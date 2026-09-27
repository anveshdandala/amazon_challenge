from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from blocking import prepare_blocking_columns
from indexing import build_token_index, filter_rare_tokens
from normalization import normalize_dataframe


def load_source(path: Path, row_limit: int | None = None) -> pd.DataFrame:
	return pd.read_csv(
		path,
		sep="\t",
		dtype={"entity_id": str},
		nrows=row_limit,
	)


def build_candidates(
	source1: pd.DataFrame,
	source23: pd.DataFrame,
	max_token_frequency: int = 200,
) -> dict[str, set[str]]:
	name_index = filter_rare_tokens(
		build_token_index(source23, "name_tokens"), max_token_frequency
	)
	address_index = filter_rare_tokens(
		build_token_index(source23, "address_tokens"), max_token_frequency
	)

	candidates: dict[str, set[str]] = {}
	for _, row in source1.iterrows():
		entity_ids: set[str] = set()
		for token in row["name_tokens"]:
			entity_ids.update(name_index.get(token, set()))
		for token in row["address_tokens"]:
			entity_ids.update(address_index.get(token, set()))
		candidates[row["entity_id"]] = entity_ids

	return candidates


def select_matches(
	source1: pd.DataFrame,
	source23: pd.DataFrame,
	candidates: dict[str, set[str]],
) -> dict[str, set[str]]:
	"""Select candidate records using normalized-field similarity.

	Exact agreement on a non-empty normalized name or address is accepted
	directly. Otherwise, a candidate must have either a highly similar name and
	a reasonably similar address, or a reasonably similar name and a highly
	similar address. Empty fields never contribute similarity evidence.
	"""
	records = source23.set_index("entity_id").to_dict("index")
	matches: dict[str, set[str]] = {}

	for _, row in source1.iterrows():
		name_tokens = row["name_tokens"]
		address_tokens = row["address_tokens"]
		matched_ids: set[str] = set()

		for entity_id in candidates[row["entity_id"]]:
			candidate = records[entity_id]
			if candidate["country"] != row["country"]:
				continue

			name = row["business_name_normalized"]
			candidate_name = candidate["business_name_normalized"]
			address = row["business_address_normalized"]
			candidate_address = candidate["business_address_normalized"]
			name_similarity = _text_similarity(name, candidate_name)
			address_similarity = _text_similarity(address, candidate_address)

			same_name = bool(name) and name == candidate_name
			same_address = bool(address) and address == candidate_address
			high_name_similarity = name_similarity >= 0.90
			reasonable_name_similarity = name_similarity >= 0.65
			high_address_similarity = address_similarity >= 0.85
			reasonable_address_similarity = address_similarity >= 0.50

			if (
				same_name
				or same_address
				or (high_name_similarity and reasonable_address_similarity)
				or (reasonable_name_similarity and high_address_similarity)
			):
				matched_ids.add(entity_id)

		matches[row["entity_id"]] = matched_ids

	return matches


def _text_similarity(left: str, right: str) -> float:
	"""Return a character-level similarity score for two normalized strings."""
	if not left or not right:
		return 0.0

	return SequenceMatcher(None, left, right).ratio()


def write_results(
	output_dir: Path,
	source1: pd.DataFrame,
	candidates: dict[str, set[str]],
	matches: dict[str, set[str]],
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)

	candidate_rows = [
		{
			"source1_entity_id": entity_id,
			"candidate_entity_ids": ",".join(sorted(candidates[entity_id])),
		}
		for entity_id in source1["entity_id"]
	]
	match_rows = [
		{
			"source1_entity_id": entity_id,
			"matched_entity_ids": ",".join(sorted(matches[entity_id])),
		}
		for entity_id in source1["entity_id"]
	]

	pd.DataFrame(candidate_rows).to_csv(
		output_dir / "candidate_pairs.tsv", sep="\t", index=False
	)
	pd.DataFrame(match_rows).to_csv(
		output_dir / "matching_results.tsv", sep="\t", index=False
	)


def run_pipeline(
    source1_path: Path,
    source2_path: Path,
    source3_path: Path,
    output_dir: Path,
    max_token_frequency: int = 200,
) -> None:

	# TEMPORARY TEST
	source1 = load_source(source1_path, row_limit=1000)
	source2 = load_source(source2_path, row_limit=10000)
	source3 = load_source(source3_path, row_limit=10000)

    # Normalize only the rows we are testing
	source1 = prepare_blocking_columns(
        normalize_dataframe(source1)
    )
	
	source2 = prepare_blocking_columns(
        normalize_dataframe(source2)
    )
	
	source3 = prepare_blocking_columns(
        normalize_dataframe(source3)
    )
	
	source23 = pd.concat(
        [source2, source3],
        ignore_index=True,
    )
	candidates = build_candidates(
        source1,
        source23,
        max_token_frequency,
    )

	matches = select_matches(
        source1,
        source23,
        candidates,
    )

	write_results(
        output_dir,
        source1,
        candidates,
        matches,
    )

    
def main() -> None:
	parser = argparse.ArgumentParser(description="Run the entity-resolution pipeline.")
	parser.add_argument("--data-dir", type=Path, default=Path("dataset/test"))
	parser.add_argument("--output-dir", type=Path, default=Path("output"))
	parser.add_argument("--max-token-frequency", type=int, default=200)
	args = parser.parse_args()

	run_pipeline(
		args.data_dir / "test_source1.tsv",
		args.data_dir / "test_source2.tsv",
		args.data_dir / "test_source3.tsv",
		args.output_dir,
		args.max_token_frequency,
	)


if __name__ == "__main__":
	main()
