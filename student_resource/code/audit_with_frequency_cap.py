import time
import polars as pl
import pandas as pd
from pathlib import Path
from blocking import prepare_blocking_columns, get_name_tokens, get_address_tokens
from normalization import normalize_dataframe
from pipeline import load_source

print("--- Step 1: Loading 10,000 S1 entities and matching ground truth ---")
t0 = time.time()
s1_raw = load_source(Path("student_resource/dataset/train/train_source1.tsv"), row_limit=10000)
s1_ids = set(s1_raw["entity_id"])

gt = pl.scan_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t").filter(
    pl.col("source1_entity_id").is_in(list(s1_ids))
).collect()

s1_norm = normalize_dataframe(s1_raw)
s1_df = prepare_blocking_columns(s1_norm)

s1_to_true_matches = {}
all_true_target_ids = set()
total_true_pairs = 0
for row in gt.iter_rows(named=True):
    s1_id = row["source1_entity_id"]
    m = row["matched_entity_ids"]
    if m:
        t_ids = set(m.split(","))
        s1_to_true_matches[s1_id] = t_ids
        all_true_target_ids.update(t_ids)
        total_true_pairs += len(t_ids)

print(f"Total true pairs: {total_true_pairs:,} across {len(all_true_target_ids):,} target IDs.")

print("\n--- Step 2: Loading true matching S2/S3 records ---")
s2_matches = pl.scan_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect().to_pandas()
s3_matches = pl.scan_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect().to_pandas()

target_df = pd.concat([s2_matches, s3_matches], ignore_index=True)
target_norm = normalize_dataframe(target_df)
target_df = prepare_blocking_columns(target_norm)
target_map = {row["entity_id"]: row for _, row in target_df.iterrows()}

print("\n--- Step 3: Computing token document frequencies on S23 ---")
# Let's count token frequencies across 1,000,000 S23 records
t0 = time.time()
s2_sample = pl.scan_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").slice(0, 500000).collect().to_pandas()
s3_sample = pl.scan_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").slice(0, 500000).collect().to_pandas()
s23_sample = pd.concat([s2_sample, s3_sample], ignore_index=True)
s23_sample = prepare_blocking_columns(normalize_dataframe(s23_sample))

name_freq = {}
addr_freq = {}
for _, row in s23_sample.iterrows():
    c = row["country"]
    for tok in row["name_tokens"]:
        k = (c, tok)
        name_freq[k] = name_freq.get(k, 0) + 1
    for tok in row["address_tokens"]:
        k = (c, tok)
        addr_freq[k] = addr_freq.get(k, 0) + 1

print(f"Token frequencies computed on 1M S23 rows in {time.time() - t0:.2f}s")

for max_freq in [50, 100, 200, 500, 1000]:
    found = 0
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        c = s1_row["country"]
        true_ids = s1_to_true_matches.get(s1_id, set())
        
        valid_name_toks = {t for t in s1_row["name_tokens"] if name_freq.get((c, t), 0) <= max_freq}
        valid_addr_toks = {t for t in s1_row["address_tokens"] if addr_freq.get((c, t), 0) <= max_freq}
        
        for tid in true_ids:
            t_row = target_map.get(tid)
            if t_row is None or t_row["country"] != c:
                continue
            if (valid_name_toks & t_row["name_tokens"]) or (valid_addr_toks & t_row["address_tokens"]):
                found += 1
                
    recall = found / total_true_pairs
    print(f"max_token_frequency = {max_freq:4d}: Recall ceiling = {recall:.4f} ({found:,} / {total_true_pairs:,})")
