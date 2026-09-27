import time
import psutil
import os
import pandas as pd
from pathlib import Path
from blocking import prepare_blocking_columns
from normalization import normalize_dataframe
from pipeline import load_source

p = psutil.Process(os.getpid())
print(f"Initial process memory: {p.memory_info().rss / 1e6:.1f} MB")

t0 = time.time()
n = 500_000
df = load_source(Path("student_resource/dataset/train/train_source2.tsv"), row_limit=n)
print(f"Loaded {n:,} rows in {time.time() - t0:.2f}s, RSS: {p.memory_info().rss / 1e6:.1f} MB")

t0 = time.time()
df_norm = normalize_dataframe(df)
print(f"Normalized {n:,} rows in {time.time() - t0:.2f}s, RSS: {p.memory_info().rss / 1e6:.1f} MB")

t0 = time.time()
df_block = prepare_blocking_columns(df_norm)
print(f"Prepared blocking columns for {n:,} rows in {time.time() - t0:.2f}s, RSS: {p.memory_info().rss / 1e6:.1f} MB")
