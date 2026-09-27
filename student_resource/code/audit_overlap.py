import time
import polars as pl
import pandas as pd
from pathlib import Path

from blocking import prepare_blocking_columns
from normalization import normalize_dataframe
from pipeline import load_source

print("--- Step 1: Loading 10,000 S1 entities and matching ground truth ---")
t0 = time.time()
s1_raw = load_source(Path("student_resource/dataset/train/train_source1.tsv"), row_limit=10000)
s1_ids = set(s1_raw["entity_id"])

# Filter ground truth to EXACTLY these 10,000 S1 IDs
gt = pl.scan_csv("student_resource/dataset/train/train_ground_truth.tsv", separator="\t").filter(
    pl.col("source1_entity_id").is_in(list(s1_ids))
).collect()

s1_norm = normalize_dataframe(s1_raw)
s1_df = prepare_blocking_columns(s1_norm)

# Collect all true matched IDs for these 10,000 S1 records
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

print(f"Loaded {len(s1_df)} S1 entities.")
print(f"Total true pairs to recover: {total_true_pairs:,} across {len(all_true_target_ids):,} unique S2/S3 IDs. ({time.time() - t0:.2f}s)")

print("\n--- Step 2: Loading true matching S2/S3 records ---")
t0 = time.time()
s2_matches = pl.scan_csv("student_resource/dataset/train/train_source2.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect().to_pandas()
s3_matches = pl.scan_csv("student_resource/dataset/train/train_source3.tsv", separator="\t").filter(
    pl.col("entity_id").is_in(list(all_true_target_ids))
).collect().to_pandas()

target_df = pd.concat([s2_matches, s3_matches], ignore_index=True)
target_norm = normalize_dataframe(target_df)
target_df = prepare_blocking_columns(target_norm)
print(f"Loaded and normalized all {len(target_df):,} true matching records in {time.time() - t0:.2f}s")

target_map = {row["entity_id"]: row for _, row in target_df.iterrows()}

print("\n--- Step 3: Checking token match without frequency cap ---")
found_name = 0
found_addr = 0
found_either_token = 0
found_prefix = 0
found_any = 0
diff_country = 0

for _, s1_row in s1_df.iterrows():
    s1_id = s1_row["entity_id"]
    true_ids = s1_to_true_matches.get(s1_id, set())
    s1_name_toks = s1_row["name_tokens"]
    s1_addr_toks = s1_row["address_tokens"]
    s1_prefix = s1_row["business_name_normalized"][:4] if s1_row["business_name_normalized"] else ""

    for tid in true_ids:
        t_row = target_map.get(tid)
        if t_row is None:
            continue
        if t_row["country"] != s1_row["country"]:
            diff_country += 1
            continue

        has_name = bool(s1_name_toks & t_row["name_tokens"])
        has_addr = bool(s1_addr_toks & t_row["address_tokens"])
        t_prefix = t_row["business_name_normalized"][:4] if t_row["business_name_normalized"] else ""
        has_prefix = bool(s1_prefix and t_prefix and s1_prefix == t_prefix)

        if has_name:
            found_name += 1
        if has_addr:
            found_addr += 1
        if has_name or has_addr:
            found_either_token += 1
        if has_prefix:
            found_prefix += 1
        if has_name or has_addr or has_prefix:
            found_any += 1

print(f"Recall from name_tokens alone:        {found_name / total_true_pairs:.4f} ({found_name}/{total_true_pairs})")
print(f"Recall from address_tokens alone:     {found_addr / total_true_pairs:.4f} ({found_addr}/{total_true_pairs})")
print(f"Recall from (name OR address tokens): {found_either_token / total_true_pairs:.4f} ({found_either_token}/{total_true_pairs})")
print(f"Recall from 4-char prefix alone:      {found_prefix / total_true_pairs:.4f} ({found_prefix}/{total_true_pairs})")
print(f"Recall from ANY (tokens OR prefix):   {found_any / total_true_pairs:.4f} ({found_any}/{total_true_pairs})")
print(f"True pairs with mismatched country:   {diff_country}")
