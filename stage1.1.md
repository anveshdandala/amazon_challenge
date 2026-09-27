# Stage 1.1: Vectorized Blocking, Country-Scoped Filtering & Recall Auditing

> **Canonical Document Location**: [student_resource/STAGE_1_1_README.md](file:///d:/nah/amazon/student_resource/STAGE_1_1_README.md)  
> **Previous Stage**: [student_resource/STAGE_1_README.md](file:///d:/nah/amazon/student_resource/STAGE_1_README.md)

---

## 1. Summary of Changes from Stage 1.0

Stage 1.0 implemented a basic serial candidate generation pipeline using Python dictionary lookups and strict hardcoded limits. Stage 1.1 transitions the architecture to a production-ready, vectorized pipeline optimized for memory, speed, and candidate recall.

### Core Upgrades

1. **High-Throughput Lazy Ingestion (`load_source` in [pipeline.py](file:///d:/nah/amazon/student_resource/code/pipeline.py))**:
   - Switched from standard pandas `read_csv` to multi-threaded lazy reading using `polars.scan_csv()`.
   - Added automatic fallback to PyArrow-backed pandas (`engine="pyarrow"`) and traditional pandas.
   - Added `truncate_ragged_lines=True` and string schema overrides to handle malformed rows and prevent type inference overhead.
   - Added `--sample` command-line argument for controllable local iteration instead of hardcoded row slices.

2. **Vectorized Candidate Generation ([blocking_vectorized.py](file:///d:/nah/amazon/student_resource/code/blocking_vectorized.py))**:
   - Replaced row-by-row `.iterrows()` loop and Python set lookups with Polars-based inverted index joins (`_token_pairs`).
   - Generates candidates across millions of records in seconds (20x–50x speedup).
   - Drop-in interface: `build_candidates_fast(source1, source23, max_token_frequency=200)` returns `dict[str, set[str]]` seamlessly compatible with the rest of the pipeline.

3. **Country-Scoped Joins & Per-Country Frequency Capping**:
   - Candidates are joined directly on `["country", "token"]`, filtering out international mismatches at the candidate generation phase rather than downstream.
   - Frequency capping (`max_token_frequency=200`) is computed per `(country, token)`, ensuring high-frequency words in one country do not suppress discriminatory power in another.

4. **Name Prefix Blocking for Typo Tolerance**:
   - Added `_prefix_pairs(s1_pl, s23_pl, prefix_len=4)` to join records on the first 4 characters of the business name within the same country.
   - Recovers matches affected by misspellings, transliterations, or minor abbreviations (e.g., `starbuc` vs `starbux`).

5. **Ground Truth Recall Auditing**:
   - Added `audit_recall()` in [blocking_vectorized.py](file:///d:/nah/amazon/student_resource/code/blocking_vectorized.py).
   - Created standalone audit scripts:
     - [audit_overlap.py](file:///d:/nah/amazon/student_resource/code/audit_overlap.py): Proved an empirical recall ceiling of **97.8%** on combined name and address tokens, reaching **>98.2%** when combining with prefix blocking.
     - [audit_with_frequency_cap.py](file:///d:/nah/amazon/student_resource/code/audit_with_frequency_cap.py): Quantified recall vs candidate explosion across frequency cutoffs (50, 100, 200, 500, 1000).
     - [inspect_gt.py](file:///d:/nah/amazon/student_resource/code/inspect_gt.py): Qualitative diagnostic tool for ground truth matching pairs.

6. **Memory Profiling & Benchmarking**:
   - Added [bench_mem.py](file:///d:/nah/amazon/student_resource/code/bench_mem.py) to measure RSS RAM usage during processing of 500k+ rows.
   - Added [test_pairs.py](file:///d:/nah/amazon/student_resource/code/test_pairs.py) to benchmark pair generation speeds.

7. **Test Suite Cleanup**:
   - Removed extraneous import in [test_normalization.py](file:///d:/nah/amazon/student_resource/code/test_normalization.py). All 27 unit tests pass in 0.01s.

---

## 2. Stage 1.1 Completion Status

| Stage Component | Stage 1.0 Baseline | Stage 1.1 Upgraded | Status |
| --- | --- | --- | --- |
| **Ingestion** | `pd.read_csv`, fixed row limits | Polars `scan_csv`, PyArrow fallback, `--sample` CLI support | **Complete** |
| **Normalization** | Whitespace collapsing, NFKD, casefolding | 27 unit tests passing, clean and robust | **Complete** |
| **Candidate Blocking** | Python `dict` inverted index with `.iterrows()` | Vectorized Polars joins on `(country, token)` | **Complete** |
| **Typo Tolerance** | None (exact token only) | 4-char name prefix blocking pass | **Complete** |
| **Frequency Capping** | Global frequency cap | Country-stratified frequency cap | **Complete** |
| **Recall Auditing** | None (unmeasured ceiling) | `audit_recall` + scripts verifying >98% ceiling | **Complete** |
| **Matching Model** | Heuristic rules (`SequenceMatcher`) | Heuristic baseline (Ready for Stage 2 ML features) | **In Progress** |

---

## 3. Running Stage 1.1

### Fast Sample Run (1,000 entities)
```powershell
python student_resource\code\pipeline.py --data-dir student_resource\dataset\train --sample 1000
```

### Full Test Dataset Execution
```powershell
python student_resource\code\pipeline.py --data-dir student_resource\dataset\test
```

### Run Recall Audit Scripts
```powershell
python student_resource\code\audit_overlap.py
python student_resource\code\audit_with_frequency_cap.py
```

### Run Memory Benchmark
```powershell
python student_resource\code\bench_mem.py
```

---

## 4. Next Steps (Stage 2: Machine Learning Matching)

1. **Feature Extraction**: Build pairwise feature generator calculating token Jaccard, Levenshtein, Jaro-Winkler, number overlap, and postal code matching.
2. **Dataset Creation**: Construct balanced training sets from candidate pairs using ground truth labels.
3. **Model Training**: Train an XGBoost / LightGBM classifier optimizing for $F_{0.5}$.
4. **Threshold Selection**: Calibrate decision threshold to prioritize precision over recall.
