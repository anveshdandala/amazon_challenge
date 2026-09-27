import re
import unicodedata
from collections.abc import Mapping
from numbers import Real
from math import isnan
from typing import Any


_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    """Create the canonical text form used for entity comparisons.

    Values are Unicode-normalized and casefolded; letters, combining marks, and
    numbers are kept, while punctuation and whitespace become separators. Missing
    values are returned as empty strings so downstream tokenization is safe.
    """
    if value is None:
        return ""

    if isinstance(value, Real) and isnan(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value)).casefold()

    cleaned = []
    for char in text:
        category = unicodedata.category(char)

        if (
            category.startswith("L")   # Letter
            or category.startswith("M")   # Combining mark
            or category.startswith("N")   # Number
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_name(value: Any) -> str:
    """Normalize a business name before name-token extraction or comparison."""
    return normalize_text(value)


def normalize_address(value: Any) -> str:
    """Normalize a business address before address-token extraction or comparison."""
    return normalize_text(value)


def normalize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one record and add normalized name and address fields when present.

    The original source fields and all unrelated fields are preserved. This is
    useful when processing a single record without mutating the input mapping.
    """
    normalized = dict(record)

    if "business_name" in normalized:
        normalized["business_name_normalized"] = normalize_name(
            normalized["business_name"]
        )

    if "business_address" in normalized:
        normalized["business_address_normalized"] = normalize_address(
            normalized["business_address"]
        )

    return normalized


def normalize_dataframe(dataframe: Any) -> Any:
    """Copy a source table and add normalized comparison columns.

    ``business_name_normalized`` and ``business_address_normalized`` are added
    only when their source columns exist. The input dataframe is left unchanged,
    making the result ready for blocking while retaining the raw source values.
    """
    normalized = dataframe.copy()

    if "business_name" in normalized.columns:
        normalized["business_name_normalized"] = normalized["business_name"].map(
            normalize_name
        )

    if "business_address" in normalized.columns:
        normalized["business_address_normalized"] = normalized["business_address"].map(
            normalize_address
        )

    return normalized