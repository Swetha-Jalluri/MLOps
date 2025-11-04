import time, argparse, itertools, logging, random
import multiprocessing as mp
from typing import Optional

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset
from transformers import AutoTokenizer


# -----------------------------
# Fixed-length rolling blocks for LM
# -----------------------------
def rolling_token_blocks(token_iter, block_size: int, pad_token_id: int, overlap: int = 0, drop_last: bool = False):
    assert 0 <= overlap < block_size
    step = block_size - overlap
    buffer = []
    for tokens in token_iter:
        buffer.extend(tokens)
        while len(buffer) >= block_size:
            chunk = buffer[:block_size]
            buffer = buffer[step:]
            yield {
                "input_ids": torch.tensor(chunk, dtype=torch.long),
                "attention_mask": torch.ones(block_size, dtype=torch.long),
            }
    if buffer and not drop_last:
        padded = buffer + [pad_token_id] * (block_size - len(buffer))
        attn = [1] * len(buffer) + [0] * (block_size - len(buffer))
        yield {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# -----------------------------
# Process-level manual sharding with resume
# -----------------------------
def manual_shard(dataset_iter, num_shards: int, process_index: int, start_index: int = 0):
    dataset_iter = itertools.islice(dataset_iter, start_index, None)
    for idx, example in enumerate(dataset_iter, start=start_index):
        if idx % num_shards == process_index:
            yield example


# -----------------------------
# Further shard by DataLoader worker
# -----------------------------
def shard_by_worker(iterable):
    wi = get_worker_info()
    if wi is None or wi.num_workers == 1:
        return iterable

    def _gen():
        for i, ex in enumerate(iterable):
            if i % wi.num_workers == wi.id:
                yield ex

    return _gen()


# -----------------------------
# Logging
# -----------------------------
def setup_logging(rank: int, log_level: str = "INFO"):
    lvl = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s [%(process)d|rank %(rank)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    class RankFilter(logging.Filter):
        def filter(self, record):
            if not hasattr(record, "rank"):
                record.rank = rank
            return True

    for h in logging.getLogger().handlers:
        h.addFilter(RankFilter())


# -----------------------------
# Dataset: only stores config; builds tokenizer/stream in __iter__
# -----------------------------
class LMStreamingDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset: str,
        dataset_config: Optional[str],
        split: str,
        text_key: str,
        model_name: str,
        block_size: int,
        overlap: int,
        drop_last: bool,
        world_size: int,
        rank: int,
        start_index: int,
        seed: int,
    ):
        self.dataset = dataset
        self.dataset_config = dataset_config
        self.split = split
        self.text_key = text_key
        self.model_name = model_name
        self.block_size = block_size
        self.overlap = overlap
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.start_index = start_index
        self.seed = seed

    def __iter__(self):
        # Tokenizer inside worker
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Fresh streaming iterator inside worker
        base_stream = load_dataset(self.dataset, self.dataset_config, split=self.split, streaming=True)

        # Validate schema once
        it = iter(base_stream)
        try:
            first = next(it)
        except StopIteration:
            raise RuntimeError("Empty dataset stream.")
        if self.text_key not in first:
            raise KeyError(f"Dataset column missing text_key='{self.text_key}'. Available keys: {list(first.keys())}")
        base_stream = itertools.chain([first], it)

        # Shard across processes, then across DataLoader workers
        base_iter = manual_shard(base_stream, self.world_size, self.rank, start_index=self.start_index)
        base_iter = shard_by_worker(base_iter)

        # Token stream per example
        token_stream = (
            tokenizer(ex[self.text_key], add_special_tokens=False)["input_ids"]
            for ex in base_iter
        )

        # Yield rolling blocks
        yield from rolling_token_blocks(
            token_stream,
            self.block_size,
            tokenizer.pad_token_id,
            overlap=self.overlap,
            drop_last=self.drop_last,
        )


# -----------------------------
# Top-level (picklable) collate
# -----------------------------
def collate_blocks(batch):
    return {
        "input_ids": torch.stack([ex["input_ids"] for ex in batch]),
        "attention_mask": torch.stack([ex["attention_mask"] for ex in batch]),
    }


# -----------------------------
# Worker entry
# -----------------------------
def worker_entry(rank: int, world_size: int, args):
    setup_logging(rank, args.log_level)
    random.seed(args.seed + rank)

    ds = LMStreamingDataset(
        dataset=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        text_key=args.text_key,
        model_name=args.model_name,
        block_size=args.block_size,
        overlap=args.overlap,
        drop_last=args.drop_last,
        world_size=world_size,
        rank=rank,
        start_index=args.start_index,
        seed=args.seed,
    )

    # Build DataLoader (Windows-safe)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        collate_fn=collate_blocks,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None),
    )

    logging.info(
        f"Starting… model={args.model_name} bs={args.batch_size} block={args.block_size} overlap={args.overlap} drop_last={args.drop_last}",
        extra={"rank": rank},
    )

    t0, toks_accum = time.time(), 0
    for i, batch in enumerate(loader):
        toks_accum += int(batch["input_ids"].numel())
        if (i + 1) % args.log_every == 0:
            dt = time.time() - t0
            tps = toks_accum / dt
            logging.info(f"batch={i+1} tokens/s={int(tps)}", extra={"rank": rank})
            t0, toks_accum = time.time(), 0

        if args.inspect > 0 and i == 0:
            tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            decoded = tok.decode(batch["input_ids"][0], skip_special_tokens=True)
            logging.info(f"peek: {decoded[:200]}…", extra={"rank": rank})

        if args.dry_run or (args.batches_total and (i + 1) >= args.batches_total):
            break

    logging.info("Done.", extra={"rank": rank})


# -----------------------------
# Launcher
# -----------------------------
def launch_multi_proc(args):
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(args.num_procs):
        p = ctx.Process(target=worker_entry, args=(rank, args.num_procs, args))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


# -----------------------------
# Args
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_procs", type=int, default=2)
    ap.add_argument("--model_name", type=str, default="gpt2")

    ap.add_argument("--dataset", type=str, default="wikitext")
    ap.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--text_key", type=str, default="text")

    ap.add_argument("--block_size", type=int, default=128)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--drop_last", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--num_workers", type=int, default=0)  # Windows + streaming: keep 0
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")

    ap.add_argument("--inspect", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--batches_total", type=int, default=0)
    ap.add_argument("--start_index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--log_level", type=str, default="INFO")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    launch_multi_proc(args)

