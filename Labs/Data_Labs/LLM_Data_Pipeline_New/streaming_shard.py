# streaming_shard.py — streaming classification with sharding, dynamic padding, and window shuffle
import os, sys, time, argparse, itertools, logging, random
from typing import Optional, Iterable, Dict, Any
import multiprocessing as mp

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from datasets import load_dataset
from transformers import AutoTokenizer

# -----------------------------
# Utility: windowed shuffle (approximate stream shuffle)
# -----------------------------
def windowed_shuffle(iterable: Iterable[Dict[str, Any]], buffer_size: int = 1024, seed: int = 42):
    rng = random.Random(seed)
    buffer = []
    it = iter(iterable)
    try:
        for _ in range(buffer_size):
            buffer.append(next(it))
    except StopIteration:
        pass
    while buffer:
        idx = rng.randrange(len(buffer))
        yield buffer.pop(idx)
        try:
            buffer.append(next(it))
        except StopIteration:
            continue

# -----------------------------
# Manual sharding with resume
# -----------------------------
def manual_shard(dataset_iter, num_shards: int, process_index: int, start_index: int = 0):
    dataset_iter = itertools.islice(dataset_iter, start_index, None)
    for idx, example in enumerate(dataset_iter, start=start_index):
        if idx % num_shards == process_index:
            yield example

# -----------------------------
# Further shard by DataLoader worker to avoid duplication
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
# Iterable dataset with optional cleaning
# -----------------------------
class TokenizedStreamingIterableDataset(IterableDataset):
    def __init__(self, dataset_iter, tokenizer, text_key: str, label_key: str, clean_text: bool = False, max_length: Optional[int] = None):
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.text_key = text_key
        self.label_key = label_key
        self.clean_text = clean_text
        self.max_length = max_length

    def _clean(self, s: str) -> str:
        if not self.clean_text:
            return s
        return " ".join(s.replace("\u200b", " ").split())

    def __iter__(self):
        base_iter = shard_by_worker(self.dataset_iter)
        for ex in base_iter:
            try:
                text = self._clean(ex[self.text_key])
                label = ex[self.label_key]
            except KeyError as e:
                missing = e.args[0]
                raise KeyError(f"Missing key '{missing}' in dataset example. "
                               f"Available keys: {list(ex.keys())}. "
                               f"Use --text_key/--label_key to set the correct columns.") from None
            toks = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            yield {
                "input_ids": toks["input_ids"].squeeze(0),
                "attention_mask": toks["attention_mask"].squeeze(0),
                "labels": torch.tensor(label, dtype=torch.long),
            }

# -----------------------------
# Dynamic padding collator with optional multiple-of
# -----------------------------
def dynamic_collate(batch, pad_id: int, pad_to_multiple_of: Optional[int] = None):
    ids = [ex["input_ids"] for ex in batch]
    attn = [ex["attention_mask"] for ex in batch]
    labels = [ex["labels"] for ex in batch]
    max_len = max(t.size(0) for t in ids)
    if pad_to_multiple_of:
        max_len = ((max_len + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

    def pad_tensor(t, value=0):
        if t.size(0) < max_len:
            pad = torch.full((max_len - t.size(0),), value, dtype=t.dtype)
            return torch.cat([t, pad])
        return t

    padded_ids = [pad_tensor(t, pad_id) for t in ids]
    padded_attn = [pad_tensor(a, 0) for a in attn]
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(padded_attn),
        "labels": torch.stack(labels),
    }

# -----------------------------
# Logging setup
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
# Worker entry
# -----------------------------
def worker_entry(rank: int, world_size: int, args):
    setup_logging(rank, args.log_level)
    random.seed(args.seed + rank)

    # Load dataset (streaming) and do a one-sample schema check
    base_stream = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    it = iter(base_stream)
    try:
        first = next(it)
    except StopIteration:
        raise RuntimeError("Empty dataset stream.")
    if args.text_key not in first or args.label_key not in first:
        raise KeyError(
            f"Dataset columns missing. Expected text_key='{args.text_key}', label_key='{args.label_key}'. "
            f"Available keys: {list(first.keys())}"
        )
    # Re-chain the first example back so nothing is lost
    base_stream = itertools.chain([first], it)

    # Manual sharding + optional windowed shuffle
    base_iter = manual_shard(base_stream, world_size, rank, start_index=args.start_index)
    if args.shuffle_buffer > 0:
        base_iter = windowed_shuffle(base_iter, buffer_size=args.shuffle_buffer, seed=args.seed + rank)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.cls_token

    # Dataset wrapper
    tokenized_iterable = TokenizedStreamingIterableDataset(
        base_iter,
        tokenizer,
        text_key=args.text_key,
        label_key=args.label_key,
        clean_text=args.clean_text,
        max_length=args.max_length,
    )

    # DataLoader
    collate = lambda b: dynamic_collate(b, pad_id=tokenizer.pad_token_id, pad_to_multiple_of=args.pad_to_multiple_of)
    loader = DataLoader(
        tokenized_iterable,
        batch_size=args.batch_size,
        collate_fn=collate,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
    )

    # Run
    logging.info(f"Starting… model={args.model_name} bs={args.batch_size}", extra={"rank": rank})
    t0 = time.time()
    seen = 0
    for i, batch in enumerate(loader):
        seen += batch["input_ids"].shape[0]
        if (i + 1) % args.log_every == 0:
            dt = time.time() - t0
            sps = (args.log_every * args.batch_size) / dt
            logging.info(f"batch={i+1} samples/s={sps:.1f} max_len={batch['input_ids'].shape[-1]}", extra={"rank": rank})
            t0 = time.time()

        if args.inspect > 0 and i == 0:
            for j in range(min(args.inspect, batch["input_ids"].shape[0])):
                decoded = tokenizer.decode(batch["input_ids"][j], skip_special_tokens=True)
                logging.info(f"sample[{j}] {decoded[:120]}…", extra={"rank": rank})

        if args.dry_run or (args.batches_total and (i + 1) >= args.batches_total):
            break

    logging.info(f"Done. seen={seen} batches={i+1}", extra={"rank": rank})

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
    ap.add_argument("--num_procs", type=int, default=4)
    ap.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    ap.add_argument("--dataset", type=str, default="ag_news")
    ap.add_argument("--dataset_config", type=str, default=None)
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--text_key", type=str, default="text")
    ap.add_argument("--label_key", type=str, default="label")

    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--prefetch_factor", type=int, default=2)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--pad_to_multiple_of", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=None, help="Tokenization max_length (None = tokenizer default)")

    ap.add_argument("--shuffle_buffer", type=int, default=0, help=">0 to enable windowed shuffle buffer")
    ap.add_argument("--start_index", type=int, default=0, help="resume index in the stream")
    ap.add_argument("--inspect", type=int, default=0, help="decode N samples from first batch")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--batches_total", type=int, default=0, help="stop after this many batches (0=unlimited)")
    ap.add_argument("--clean_text", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--log_level", type=str, default="INFO")
    ap.add_argument("--chaos", action="store_true")

    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    launch_multi_proc(args)
