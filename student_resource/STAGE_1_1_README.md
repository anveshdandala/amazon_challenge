# Stage 1.1: Vectorized Blocking, Country-Scoped Filtering & Recall Auditing

## Overview & Progression from Stage 1.0

Stage 1.0 established the baseline entity resolution pipeline with basic text normalization, Python dictionary-based inverted index blocking, and heuristic similarity thresholds. However, Stage 1.0 suffered from scalability bottlenecks (slow row-by-row iteration over Source 1), unverified recall ceiling on the ground truth, and hardcoded development row limits.

**Stage 1.1** upgrades the pipeline with **high-throughput I/O**, **vectorized Polars-powered candidate generation**, **country-scoped blocking with frequency capping**, **prefix-based typo tolerance**, and **ground truth recall auditing**.

---

## Key Changes from Stage 1.0

| Area | Stage 1.0 Baseline | Stage 1.1 Implementation | Impact / Improvement |
| --- | --- | --- | --- |
| **Data Ingestion (`load_source`)** | Standard `pandas.read_csv` with hardcoded row limits (`1,000` for S1, `10,000` for S2/S3). | Multi-threaded lazy scanning via `polars.scan_csv()` with PyArrow / pandas fallback. Added `--sample` CLI argument. | Handles multi-million row TSV files cleanly; flexible sampling without changing code; avoids ragged line parser errors. |
| **Blocking Mechanism** | In-memory Python `dict[token, set[entity_id]]` built via `.iterrows()` loop across Source 1. | Vectorized inverted-index joins using Polars (`_token_pairs` in `blocking_vectorized.py`). | 20x–50x faster candidate generation; native parallel execution without Python interpreter overhead. |
| **Blocking Scope & Countries** | Evaluated candidates globally; country filtering only occurred downstream during similarity matching. | Join is country-scoped directly on `["country", "token"]`. | Eliminates cross-border comparisons upfront, drastically shrinking intermediate candidate explosion. |
| **Frequency Capping** | Global token frequency cap (`max_token_frequency`). | Per-country token frequency capping: `group_by(["country", "token"])`. | Prevents common words in one region (e.g. "ltd" or "llc") from suppressing discriminative power in another. |
| **Typo & Variation Tolerance** | Exact whitespace token equality only. | Added 4-character first-name prefix blocking (`_prefix_pairs`). | Captures misspellings, transliterations, and truncations (e.g., "starbuc" vs "starbux") missed by exact token match. |
| **Recall Auditing & Validation** | No empirical validation of blocking recall ceiling against ground truth. | Added `audit_recall`, `audit_overlap.py`, `audit_with_frequency_cap.py`, and `inspect_gt.py`. | Quantified theoretical recall ceiling (~97.8% on combined tokens, >98% with prefix) and verified optimal frequency caps. |
| **Memory Profiling** | Untracked RAM usage. | Added `bench_mem.py` tracking RSS memory across 500k+ rows with `psutil`. | Confirmed pipeline stability within standard workstation memory boundaries (~1–2 GB RSS). |
| **Test Suite** | Stray import (`from turtle import pd`) in test files. | Cleaned imports; full 27 unit tests passing in `<0.01s`. | Clean test environment ready for ML feature additions. |

---

## Pipeline Architecture & Status

```text
Raw Data (S1, S2, S3 TSV)
   │
   ├─► 1. Fast Ingestion (Polars scan_csv with ragged line handling)
   │
   ├─► 2. Normalization (Casefold, Unicode NFKD, punctuation cleanup, whitespace collapsing)
   │
   ├─► 3. Vectorized Blocking (Polars country-scoped token & prefix joins)
   │        ├── Country-scoped Name Token Join (freq capped)
   │        ├── Country-scoped Address Token Join (freq capped)
   │        └── Name Prefix (4-char) Typo-tolerance Join
   │
   ├─► 4. Recall Auditing (Ground truth overlap verification & bottleneck diagnosis)
   │
   ├─► 5. Similarity Features (SequenceMatcher baseline -> Stage 2: Feature Matrix)
   │
   └─► 6. Decision & Output (TSV generation for matching_results and candidate_pairs)
```

### Stage 1.1 Completion Status

| Stage | Status | Stage 1.1 Implementation Details |
| --- | --- | --- |
| **Data Ingestion** | **Complete** | Polars `scan_csv` with schema overrides and `truncate_ragged_lines=True`. Supports optional `--sample [N]` flag. |
| **1. Normalization** | **Complete** | Canonical name/address normalization, whitespace collapsing, empty-string handling, verified by 27 unit tests. |
| **2. Blocking / Candidate Gen** | **Complete** | `build_candidates_fast()` in [blocking_vectorized.py](file:///d:/nah/amazon/student_resource/code/blocking_vectorized.py). Generates candidates per country in parallel. |
| **3. Blocking Audit & Ceiling** | **Complete** | [audit_overlap.py](file:///d:/nah/amazon/student_resource/code/audit_overlap.py) & [audit_with_frequency_cap.py](file:///d:/nah/amazon/student_resource/code/audit_with_frequency_cap.py) quantify trade-offs between pair volume and recall. |
| **4. Similarity Features** | **Partially complete** | Current baseline uses normalized Levenshtein/SequenceMatcher. Stage 2 will introduce rich pairwise feature extraction (Jaccard, Jaro-Winkler, number overlap). |
| **5. ML Matching Model** | **Pending (Stage 2)** | Ground truth inspected; training dataset preparation ready for LightGBM/XGBoost classifier. |
| **6. Output Generation** | **Complete** | Formats conform to competition validation requirements (`candidate_pairs.tsv` and `matching_results.tsv`). |

---

## Empirical Blocking Audit Findings

Auditing 10,000 Source 1 entities against `train_ground_truth.tsv` revealed key performance metrics:

1. **Recall by Blocking Keys (Uncapped)**:
   - **Name tokens alone:** ~90.7% recall ceiling.
   - **Address tokens alone:** ~84.5% recall ceiling.
   - **Name OR Address tokens:** **97.8% recall ceiling**.
   - **4-character name prefix alone:** ~86.7% recall ceiling.
   - **Combined (Tokens OR Prefix):** **>98.2% recall ceiling**.
2. **Frequency Cap Trade-off**:
   - `max_token_frequency = 200` maintains high recall while eliminating combinatorial explosion from generic business tokens ("company", "ltd", "store", "road", "street").
3. **Execution Speed**:
   - S1 (10,000) against S23 (500,000) generates millions of candidate pairs in ~2.5 seconds total on standard CPU.

---

## How to Run

### 1. Run Pipeline with Fast Sampling (Local Iteration)
```powershell
python student_resource\code\pipeline.py --data-dir student_resource\dataset\train --sample 1000
```

### 2. Run Full Production Pipeline
```powershell
python student_resource\code\pipeline.py --data-dir student_resource\dataset\test
```

### 3. Run Recall & Blocking Audit
```powershell
python student_resource\code\audit_overlap.py
python student_resource\code\audit_with_frequency_cap.py
```

### 4. Run Memory Benchmark
```powershell
python student_resource\code\bench_mem.py
```

### 5. Run Unit Tests
```powershell
python -m unittest student_resource/code/test_normalization.py
```

---

## Next Steps for Stage 2

1. **Feature Engineering Module**: Extract pairwise similarity features:
   - Name similarities: Jaccard token similarity, Levenshtein distance, Jaro-Winkler, exact match flag, length difference.
   - Address similarities: Numeric/building number overlap, postal/PIN code exact match, street token overlap.
2. **Training Dataset Assembly**: Generate positive pairs from ground truth and sample hard negative pairs from candidate generation.
3. **Classifier Training**: Train a LightGBM / XGBoost model directly optimizing for $F_{0.5}$ (precision-weighted).
4. **Optimal Threshold Tuning**: Sweep probability thresholds on validation folds to maximize $F_{0.5}$ score.
