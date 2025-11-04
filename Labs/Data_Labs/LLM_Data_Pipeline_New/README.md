# LLM Data Pipeline — Labs (Streaming Datasets)

This repository contains **two hands-on labs** demonstrating *streaming data pipelines* using Hugging Face `datasets` and PyTorch `IterableDataset`.

- **Lab 1:** Streaming **Text Classification** with AG News  
- **Lab 2:** Streaming **Language Modeling** with WikiText-2  

These labs demonstrate how to build scalable, efficient, and distributed input pipelines that can process large language datasets **without full downloads**, focusing on real-world streaming, sharding, and batching mechanisms.

---

## Overview

You will explore:
- True dataset streaming from Hugging Face Hub  
- Tokenization and batching on the fly  
- Dynamic padding and shuffle buffering  
- Process-level and worker-level sharding  
- Overlapping rolling token blocks for LM  
- Multi-processing for distributed streaming  
- Visualization and inspection utilities for performance

---

## Files Included

```
streaming_shard.py         # Lab 1: Text Classification
streaming_lm_shard.py      # Lab 2: Language Modeling
Lab1.ipynb                 # Notebook version of Lab 1
Lab 2.ipynb                # Notebook version of Lab 2
requirements.txt           # Dependencies list
README.md                  # This documentation
```

---

## Environment Setup

**Recommended Python version:** 3.10 or higher

```bash
python -m venv .venv

# Activate virtual environment
# Windows
. .venv/Scripts/Activate.ps1
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install torch datasets transformers matplotlib tqdm
```

> **Windows Note:**  
> Hugging Face may warn about symlink caching — this is harmless.  
> Enable *Developer Mode* to remove the warning.

---

## Lab 1 — Streaming Text Classification (AG News)

**File:** `streaming_shard.py`

This lab focuses on streaming-based text classification using the AG News dataset.

### Concepts Covered
- Online streaming from Hugging Face (no local cache)
- Dynamic tokenization per sample
- Windowed shuffle buffering for streaming randomness
- Batch-level dynamic padding and collation
- Multi-process sharding for parallel streaming
- Inspection and throughput logging

### Key Features Implemented
1. **Windowed Shuffle Buffer:** Randomizes the order of streamed data using a buffer instead of full dataset shuffling.  
2. **Dynamic Padding:** Automatically pads each batch to the longest sequence to minimize wasted computation.  
3. **Dynamic Collate Function:** Efficient collation and padding using `torch.utils.data.DataLoader`.  
4. **Optional Text Cleaning:** Added `--clean_text` flag to remove hidden/invisible characters before tokenization.  
5. **Cross-process Sharding:** Data is evenly split across processes and workers with `manual_shard()` and `shard_by_worker()`.  
6. **Real-time Throughput Logging:** Periodic display of processed batches, token counts, and samples/sec.  
7. **Sample Inspection:** `--inspect` argument prints decoded examples from the stream for quick validation.  
8. **Parameter Flexibility:** Almost every aspect (buffer size, padding multiple, batch size, model name) is configurable via command line.

### Example Run (Windows-safe)
```bash
python streaming_shard.py ^
  --num_procs 2 ^
  --num_workers 0 ^
  --batches_total 10 ^
  --inspect 2
```

### Common Options
```
--model_name distilbert-base-uncased
--dataset ag_news --split train
--batch_size 8
--pad_to_multiple_of 8
--shuffle_buffer 1024
--max_length 256
--log_every 5
--inspect 2
```

---

## Lab 2 — Streaming Language Modeling (WikiText-2)

**File:** `streaming_lm_shard.py`

This lab demonstrates a streaming pipeline for autoregressive training using GPT-2 and the WikiText-2 dataset.

### Concepts Covered
- Continuous text streaming from Hugging Face
- Tokenized block generation for LM
- Overlapping token blocks for context continuity
- Process-level sharding for scalable parallelism
- EOS-based padding for GPT models
- Real-time throughput monitoring

### Key Features Implemented
1. **Rolling Token Blocks:** Token sequences are grouped into overlapping fixed-length blocks for consistent context size.  
2. **Overlap Control:** Added `--overlap` parameter for adjustable overlap between rolling windows.  
3. **EOS Padding for GPT-2:** GPT-2 tokenizer patched to use its EOS token for consistent block padding.  
4. **Drop-Last Option:** Added `--drop_last` to ignore incomplete token blocks.  
5. **Multi-Process Sharded Streaming:** Ensures each process receives a unique subset of streamed samples.  
6. **Throughput and Progress Logging:** Reports number of processed tokens and throughput periodically.  
7. **Peek and Inspect:** `--inspect 1` displays the first few decoded token blocks for visual verification.

### Example Run (Windows-safe)
```bash
python streaming_lm_shard.py ^
  --num_procs 2 ^
  --num_workers 0 ^
  --batches_total 5 ^
  --inspect 1
```

### Common Options
```
--model_name gpt2
--dataset wikitext --dataset_config wikitext-2-raw-v1 --split train
--block_size 128
--overlap 32
--batch_size 4
--log_every 2
--inspect 1
```

---

## Why Use `--num_workers 0` on Windows

- Windows multiprocessing uses `spawn` instead of `fork`.  
- Tokenizer objects and dataset generators cannot be pickled easily.  
- To ensure safe parallelization, use `--num_workers 0`.  
- On Linux/macOS, higher worker counts (e.g., `--num_workers 2`) can be safely used.

---

## Sharding Logic

### Process-level Sharding
Each process reads every *N-th* sample in the stream, ensuring that data is evenly split between parallel workers.

Example:
- Process 0 → samples 0, 2, 4, …  
- Process 1 → samples 1, 3, 5, …

### Worker-level Sharding
If DataLoader workers > 1, each worker receives a disjoint subset of that process’s stream to avoid duplication.

This ensures **no overlap** and **no skipped samples**.

---

## Inspecting Samples

Use `--inspect N` to decode and print the first `N` samples or token blocks.  
Helps confirm:
- Dataset field names (`--text_key`, `--label_key`)
- Tokenizer configuration and truncation
- Padding correctness
- Overlap logic for LM tasks

---

## Troubleshooting

- **Pin memory warning:**  
  `'pin_memory' ... no accelerator found` — safe to ignore when using CPU.

- **Symlink warning (Windows):**  
  Harmless — ignore or enable Developer Mode.

- **Pickling errors:**  
  Use `--num_workers 0`.

- **Duplicate samples warning:**  
  Occurs when workers exceed number of shards — lower `num_workers`.

---

## Suggested `requirements.txt`

```
torch>=2.1
datasets>=2.19
transformers>=4.41
matplotlib>=3.7
tqdm>=4.66
```

---

## License

You may use any permissive open-source license (e.g., MIT).

