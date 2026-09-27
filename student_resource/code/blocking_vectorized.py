"""
Vectorized (join-based) blocking — drop-in replacement for
`build_token_index` / `filter_rare_tokens` / `build_candidates`.

Reuses the exact tokenization logic from blocking.py (get_name_tokens /
get_address_tokens via prepare_blocking_columns) — only the candidate
*lookup* mechanism changes, from Python dict-of-sets built via iterrows to
a polars join. Behavior should match the original pipeline closely; the
main functional addition is an optional prefix-blocking pass for typo
tolerance, and country-scoped joins to shrink join size.

Usage (drop-in for pipeline.py):

    from blocking_vectorized import build_candidates_fast

    source1 = prepare_blocking_columns(normalize_dataframe(source1))
    source23 = prepare_blocking_columns(normalize_dataframe(source23))

    candidates = build_candidates_fast(source1, source23, max_token_frequency=200)
    # candidates: dict[str, set[str]] -- same shape as original build_candidates
"""

from __future__ import annotations

import polars as pl


def _to_polars(df, token_cols=("name_tokens", "address_tokens")):
    """Pandas df with set-valued token columns -> polars df with list columns.

    polars can't ingest Python `set` objects directly, so we convert to
    lists first (cheap, linear pass -- not the bottleneck).
    """
    pdf = df[["entity_id", "country", *token_cols]].copy()
    for col in token_cols:
        pdf[col] = pdf[col].map(list)
    return pl.from_pandas(pdf)


def _token_pairs(
    s1_pl: pl.DataFrame,
    s23_pl: pl.DataFrame,
    token_col: str,
    max_token_frequency: int,
) -> pl.DataFrame:
    """Inverted-index join for one token column, scoped to matching country."""
    s1 = (
        s1_pl.select(["entity_id", "country", token_col])
        .explode(token_col)
        .rename({"entity_id": "source1_entity_id", token_col: "token"})
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
    )
    s23 = (
        s23_pl.select(["entity_id", "country", token_col])
        .explode(token_col)
        .rename({token_col: "token"})
        .filter(pl.col("token").is_not_null() & (pl.col("token") != ""))
    )

    # frequency cap -- mirrors filter_rare_tokens, computed per (country, token)
    # so a token common in the US doesn't suppress it as a signal in India.
    freq = s23.group_by(["country", "token"]).agg(pl.len().alias("freq"))
    keep = freq.filter(pl.col("freq") <= max_token_frequency).drop("freq")
    s23 = s23.join(keep, on=["country", "token"], how="inner")

    return (
        s1.join(s23, on=["country", "token"], how="inner")
        .select(["source1_entity_id", "entity_id"])
        .unique()
    )


def _prefix_pairs(
    s1_pl: pl.DataFrame, s23_pl: pl.DataFrame, prefix_len: int = 4
) -> pl.DataFrame:
    """First-name-token prefix blocking -- catches typos/transliteration that
    defeat exact token matching (e.g. 'starbuc' vs 'starbux' share 'star').
    Only fires when a name token exists at all.
    """

    def prep(df, id_out):
        return (
            df.select(["entity_id", "country", "name_tokens"])
            .filter(pl.col("name_tokens").list.len() > 0)
            .with_columns(
                pl.col("name_tokens").list.first().str.slice(0, prefix_len).alias("prefix")
            )
            .rename({"entity_id": id_out} if id_out else {})
            .select([id_out or "entity_id", "country", "prefix"])
        )

    s1 = prep(s1_pl, "source1_entity_id")
    s23 = prep(s23_pl, "entity_id")

    return s1.join(s23, on=["country", "prefix"], how="inner").select(
        ["source1_entity_id", "entity_id"]
    ).unique()


def build_candidates_fast(
    source1_pd,
    source23_pd,
    max_token_frequency: int = 200,
    use_name_prefix_block: bool = True,
    prefix_len: int = 4,
) -> dict[str, set[str]]:
    """Vectorized replacement for build_token_index+filter_rare_tokens+build_candidates.

    source1_pd / source23_pd: pandas DataFrames after prepare_blocking_columns
    (must have entity_id, country, name_tokens, address_tokens columns).

    Returns dict[source1_entity_id, set(candidate_entity_ids)] -- same shape
    the rest of pipeline.py (select_matches, write_results) already expects.
    """
    s1_pl = _to_polars(source1_pd)
    s23_pl = _to_polars(source23_pd)

    name_pairs = _token_pairs(s1_pl, s23_pl, "name_tokens", max_token_frequency)
    addr_pairs = _token_pairs(s1_pl, s23_pl, "address_tokens", max_token_frequency)
    all_pairs = pl.concat([name_pairs, addr_pairs])

    if use_name_prefix_block:
        all_pairs = pl.concat([all_pairs, _prefix_pairs(s1_pl, s23_pl, prefix_len)])

    all_pairs = all_pairs.unique()

    grouped = all_pairs.group_by("source1_entity_id").agg(
        pl.col("entity_id").alias("candidates")
    )

    result: dict[str, set[str]] = {sid: set() for sid in source1_pd["entity_id"]}
    for row in grouped.iter_rows(named=True):
        result[row["source1_entity_id"]] = set(row["candidates"])
    return result


def audit_recall(candidates: dict[str, set[str]], ground_truth_pd) -> dict:
    """Run on TRAINING data before writing model code.

    ground_truth_pd: pandas df with source1_entity_id, matched_entity_ids
    (comma-separated string), i.e. train_ground_truth.tsv loaded as-is.

    Returns the recall ceiling your blocking imposes, plus which entities
    are missed -- so you know whether to loosen max_token_frequency, add
    another blocking key, or investigate a specific pattern/country.
    """
    total_true_pairs = 0
    found_pairs = 0
    missed_by_entity: dict[str, int] = {}

    for _, row in ground_truth_pd.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = row["matched_entity_ids"]
        if not isinstance(matched_str, str) or not matched_str:
            continue
        true_ids = set(matched_str.split(","))
        total_true_pairs += len(true_ids)

        cand = candidates.get(s1_id, set())
        found = true_ids & cand
        found_pairs += len(found)

        missed = len(true_ids) - len(found)
        if missed:
            missed_by_entity[s1_id] = missed

    recall_ceiling = found_pairs / total_true_pairs if total_true_pairs else 1.0
    worst = sorted(missed_by_entity.items(), key=lambda kv: kv[1], reverse=True)[:20]

    return {
        "total_true_pairs": total_true_pairs,
        "found_pairs": found_pairs,
        "recall_ceiling": recall_ceiling,
        "num_entities_with_misses": len(missed_by_entity),
        "worst_entities": worst,
    }