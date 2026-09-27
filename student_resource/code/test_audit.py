import time
import pandas as pd
from pathlib import Path

from blocking import prepare_blocking_columns
from normalization import normalize_dataframe
from pipeline import load_source
from blocking_vectorized import build_candidates_fast, audit_recall

data_dir = Path("student_resource/dataset/train")
s1_path = data_dir / "train_source1.tsv"
s2_path = data_dir / "train_source2.tsv"
s3_path = data_dir / "train_source3.tsv"
gt_path = data_dir / "train_ground_truth.tsv"

print("Testing audit_recall on sample=1000...")
t0 = time.time()
s1 = prepare_blocking_columns(normalize_dataframe(load_source(s1_path, row_limit=1000)))
s2 = prepare_blocking_columns(normalize_dataframe(load_source(s2_path, row_limit=5000)))
s3 = prepare_blocking_columns(normalize_dataframe(load_source(s3_path, row_limit=5000)))
s23 = pd.concat([s2, s3], ignore_index=True)
print(f"Loaded and normalized in {time.time() - t0:.2f}s")

t0 = time.time()
candidates = build_candidates_fast(s1, s23, max_token_frequency=200)
print(f"Candidates built in {time.time() - t0:.2f}s")

avg_cands = sum(len(c) for c in candidates.values()) / len(candidates)
print(f"Average candidates per S1: {avg_cands:.1f}")

gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False, nrows=1000)
report = audit_recall(candidates, gt)
print("Recall ceiling:", report["recall_ceiling"], f"({report['found_pairs']} / {report['total_true_pairs']})")
