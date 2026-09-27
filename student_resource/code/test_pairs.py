import time
import polars as pl
from pathlib import Path
from blocking import prepare_blocking_columns
from normalization import normalize_dataframe
from pipeline import load_source
from blocking_vectorized import _to_polars, _token_pairs, _prefix_pairs

print("Loading 10,000 S1 and 500,000 S23...")
s1 = prepare_blocking_columns(normalize_dataframe(load_source(Path("student_resource/dataset/train/train_source1.tsv"), row_limit=10000)))
s2 = prepare_blocking_columns(normalize_dataframe(load_source(Path("student_resource/dataset/train/train_source2.tsv"), row_limit=250000)))
s3 = prepare_blocking_columns(normalize_dataframe(load_source(Path("student_resource/dataset/train/train_source3.tsv"), row_limit=250000)))
import pandas as pd
s23 = pd.concat([s2, s3], ignore_index=True)

s1_pl = _to_polars(s1)
s23_pl = _to_polars(s23)

t0 = time.time()
name_pairs = _token_pairs(s1_pl, s23_pl, "name_tokens", max_token_frequency=200)
print(f"name_pairs: {len(name_pairs):,} in {time.time() - t0:.2f}s")

t0 = time.time()
addr_pairs = _token_pairs(s1_pl, s23_pl, "address_tokens", max_token_frequency=200)
print(f"addr_pairs: {len(addr_pairs):,} in {time.time() - t0:.2f}s")

t0 = time.time()
pref_pairs = _prefix_pairs(s1_pl, s23_pl, prefix_len=4)
print(f"pref_pairs: {len(pref_pairs):,} in {time.time() - t0:.2f}s")
