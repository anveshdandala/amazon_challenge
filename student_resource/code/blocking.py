from __future__ import annotations

from collections import defaultdict, Counter
from typing import Any


COMMON_BUSINESS_WORDS = {
    "inc",
    "incorporated",
    "ltd",
    "limited",
    "llc",
    "corp",
    "corporation",
    "company",
    "co",
    "private",
    "pvt",
    "plc",
    "llp",
}


def get_name_tokens(text: str) -> set[str]:
    """Return meaningful tokens from a normalized business name.

    Legal suffixes and other common business words are removed because they
    occur in many unrelated records. Tokens shorter than two characters are also
    ignored. The resulting set is used to create name-based blocking keys.
    For example, ``"acme corporation"`` becomes ``{"acme"}``.
    """
    if not text:
        return set()

    return {
        token
        for token in text.split()
        if token not in COMMON_BUSINESS_WORDS
        and len(token) >= 2
    }


def get_address_tokens(text: str) -> set[str]:
    """Return address tokens suitable for blocking candidate records.

    The normalized address is split on whitespace and tokens shorter than two
    characters are discarded. Unlike name tokenization, common address words
    are retained because they can help distinguish locations. For example,
    ``"123 main street"`` becomes ``{"123", "main", "street"}``.
    """

    if not text:
        return set()
    
    return {
    token
    for token in text.split()
    if len(token) >= 2
    }

def prepare_blocking_columns(dataframe):
    """Copy a normalized dataframe and add set-valued blocking columns.

    Each row receives ``name_tokens`` and ``address_tokens`` derived from the
    corresponding normalized text columns. These columns let the indexing and
    candidate-generation stages compare shared tokens instead of every possible
    source-1/source-2/source-3 pair.
    """

    result = dataframe.copy()

    result["name_tokens"] = result[
        "business_name_normalized"
    ].map(get_name_tokens)

    result["address_tokens"] = result[
        "business_address_normalized"
    ].map(get_address_tokens)

    return result