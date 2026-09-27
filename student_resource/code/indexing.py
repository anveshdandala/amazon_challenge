from __future__ import annotations

from collections import defaultdict


def build_token_index(
    dataframe,
    token_column: str,
    id_column: str = "entity_id",
) -> dict[str, set[str]]:
    """Map each token to the entity IDs of rows containing that token.

    ``dataframe`` must contain a token-set column and an ID column. The inverted
    index supports fast blocking lookups: given a source-1 token, the matcher can
    retrieve only source-2/source-3 entities that share it instead of scanning
    the full reference table. Each entity ID appears at most once per token.
    """
    index: dict[str, set[str]] = defaultdict(set)

    for _, row in dataframe.iterrows():
        entity_id = row[id_column]

        for token in row[token_column]:
            index[token].add(entity_id)

    return dict(index)


def filter_rare_tokens(
    token_index: dict[str, set[str]],
    max_frequency: int = 200,
) -> dict[str, set[str]]:
    """Keep tokens whose document frequency is within the blocking limit.

    Very common tokens produce large, low-quality candidate sets, so tokens that
    occur in more than ``max_frequency`` distinct entities are removed. The
    returned mapping is used as the filtered lookup index during candidate
    generation; the input mapping is not modified.
    """
    return {
        token: entity_ids
        for token, entity_ids in token_index.items()
        if len(entity_ids) <= max_frequency
    }