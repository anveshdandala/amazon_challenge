# Stage 1: Entity Resolution Pipeline

## Objective

The project resolves records that describe the same business across three data sources:

- Source 1: reference entities to match
- Source 2 and Source 3: records that may refer to Source 1 entities

The intended processing flow is:

```text
Raw data
   |
1. Normalization
   |
2. Blocking / Candidate Generation
   |
3. Similarity Features
   |
4. ML Matching Model
   |
5. Threshold / Decision
   |
6. Final matched IDs
```

## Stage 1 Completion Status

| Stage | Status | Current implementation |
| --- | --- | --- |
| Raw data | Complete | `load_source()` reads the tab-separated source files and preserves `entity_id` as text. |
| 1. Normalization | Complete | `normalize_dataframe()` creates canonical name and address fields. Text is casefolded, Unicode-normalized, punctuation-safe, and whitespace-collapsed. Missing values become empty strings. |
| 2. Blocking / Candidate Generation | Complete | `prepare_blocking_columns()` creates name and address token sets. `build_token_index()` and `filter_rare_tokens()` build filtered lookup indexes, and `build_candidates()` returns plausible Source 2/3 records for each Source 1 record. |
| 3. Similarity Features | Partially complete | `_text_similarity()` calculates character-level `SequenceMatcher` scores for normalized names and addresses inside `select_matches()`. A separate feature table is not implemented yet. |
| 4. ML Matching Model | Not started | The current matcher is rule-based. No trained ML model is used yet. |
| 5. Threshold / Decision | Partially complete | `select_matches()` applies exact-match rules and hand-written similarity thresholds: high name plus reasonable address, or reasonable name plus high address. Country must also agree. |
| 6. Final matched IDs | Complete for the current run | `write_results()` writes `matching_results.tsv` and `candidate_pairs.tsv`. Final matches are selected from the generated candidate set. |

## What Is Completed So Far

The current pipeline can:

1. Read Source 1, Source 2, and Source 3 TSV files.
2. Normalize business names and addresses without changing the raw columns.
3. Generate token-based candidate pairs instead of comparing every possible record pair.
4. Compare candidate records using normalized name and address similarity.
5. Exclude candidates from a different country.
6. Write candidate pairs and final matched IDs in the required TSV format.

## Current Matching Logic

A candidate is selected when the countries agree and at least one of these conditions is true:

- The normalized names are exactly equal and non-empty.
- The normalized addresses are exactly equal and non-empty.
- Name similarity is at least `0.90` and address similarity is at least `0.50`.
- Name similarity is at least `0.65` and address similarity is at least `0.85`.

Empty name or address values do not provide similarity evidence.

These thresholds are a temporary baseline. The planned next step is to expose the name and address scores, plus additional comparison features, as training inputs for an ML matching model. The model will then replace the hand-written decision rules.

## Running Stage 1

From the repository root:

```powershell
python student_resource\code\pipeline.py --data-dir student_resource\dataset\test
```

The command writes:

```text
output/candidate_pairs.tsv
output/matching_results.tsv
```

The current development run intentionally limits processing to the first 1,000 Source 1 rows and first 10,000 rows from each of Source 2 and Source 3. This keeps iteration practical while the baseline is being developed; it is not yet a full-data production run.

## Next Development Steps

1. Remove or configure the temporary row limits for full-dataset processing.
2. Create an explicit similarity-feature table for every candidate pair.
3. Use `train_ground_truth.tsv` to train and validate an ML matching model.
4. Tune the decision threshold against the precision-heavy `F_0.5` evaluation metric.
5. Run the final pipeline over every Source 1 test entity and validate both output files before submission.
