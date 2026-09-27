from turtle import pd
import unittest
import pandas as pd

from normalization import normalize_address, normalize_name, normalize_record, normalize_dataframe


class NormalizationTests(unittest.TestCase):
    def test_name_punctuation_becomes_a_separator(self):
        self.assertEqual(
            normalize_name("Orelee's Barbershop"),
            "orelee s barbershop",
        )

    def test_address_formatting_is_canonicalized(self):
        self.assertEqual(
            normalize_address(" 1795 Westchester Drive, High Point, NC "),
            "1795 westchester drive high point nc",
        )

    def test_unicode_text_is_preserved_and_casefolded(self):
        self.assertEqual(normalize_name("राम मार्केटिंग"), "राम मार्केटिंग")

    def test_missing_values_are_empty(self):
        self.assertEqual(normalize_name(None), "")
        self.assertEqual(normalize_address(float("nan")), "")

    def test_record_adds_normalized_fields_without_overwriting_source(self):
        record = {
            "business_name": "Orelee's Barbershop",
            "business_address": "1795 Westchester Drive, High Point, NC",
        }
        normalized = normalize_record(record)
        self.assertEqual(normalized["business_name"], record["business_name"])
        self.assertEqual(
            normalized["business_name_normalized"], "orelee s barbershop"
        )
        self.assertEqual(
            normalized["business_address_normalized"],
            "1795 westchester drive high point nc",
        )
    
    def test_unicode_text_is_preserved_and_casefolded(self):
        self.assertEqual(
            normalize_name("राम मार्केटिंग"),
            "राम मार्केटिंग",
        )


    def test_name_is_casefolded(self):
        self.assertEqual(
            normalize_name("ACME CORPORATION"),
            "acme corporation",
        )


    def test_mixed_case_is_normalized(self):
        self.assertEqual(
            normalize_name("McDonald's Coffee SHOP"),
            "mcdonald s coffee shop",
        )


    def test_punctuation_becomes_separator(self):
        self.assertEqual(
            normalize_name("Orelee's Barbershop"),
            "orelee s barbershop",
        )


    def test_multiple_punctuation_marks(self):
        self.assertEqual(
            normalize_name("A.B.C. Pvt. Ltd."),
            "a b c pvt ltd",
        )


    def test_extra_whitespace_is_collapsed(self):
        self.assertEqual(
            normalize_name("  ABC    Business   Center  "),
            "abc business center",
        )


    def test_tabs_and_newlines_are_normalized(self):
        self.assertEqual(
            normalize_name("ABC\tBusiness\nCenter"),
            "abc business center",
        )


    def test_unicode_composed_characters_are_normalized(self):
        self.assertEqual(
            normalize_name("Café"),
            "café",
        )


    def test_unicode_symbols_are_separators(self):
        self.assertEqual(
            normalize_name("R+ Vernal Inc"),
            "r vernal inc",
        )


    def test_numbers_are_preserved(self):
        self.assertEqual(
            normalize_name("7-Eleven"),
            "7 eleven",
        )


    def test_alphanumeric_business_name_is_preserved(self):
        self.assertEqual(
            normalize_name("3M Technologies"),
            "3m technologies",
        )


    def test_none_becomes_empty_string(self):
        self.assertEqual(
            normalize_name(None),
            "",
        )


    def test_nan_becomes_empty_string(self):
        self.assertEqual(
            normalize_name(float("nan")),
            "",
        )


    def test_empty_string_remains_empty(self):
        self.assertEqual(
            normalize_name(""),
            "",
        )


    def test_whitespace_only_becomes_empty(self):
        self.assertEqual(
            normalize_name("     "),
            "",
        )


    def test_address_is_normalized(self):
        self.assertEqual(
            normalize_address("123 Main Street, New York"),
            "123 main street new york",
        )


    def test_address_abbreviation_is_not_expanded(self):
        self.assertEqual(
            normalize_address("123 Main St."),
            "123 main st",
        )


    def test_name_and_address_normalization_are_consistent(self):
        value = "ABC, Business & Co."
        self.assertEqual(
            normalize_name(value),
            normalize_address(value),
        )


    def test_normalize_record_adds_normalized_fields(self):
        record = {
            "entity_id": "123",
            "business_name": "ACME, Inc.",
            "business_address": "12 Main St.",
            "country": "US",
        }

        result = normalize_record(record)

        self.assertEqual(result["business_name_normalized"], "acme inc")
        self.assertEqual(result["business_address_normalized"], "12 main st")
        self.assertEqual(result["entity_id"], "123")
        self.assertEqual(result["country"], "US")


    def test_normalize_record_does_not_modify_original_record(self):
        record = {
            "business_name": "ACME, Inc.",
            "business_address": "12 Main St.",
        }

        result = normalize_record(record)

        self.assertNotEqual(id(record), id(result))
        self.assertNotIn("business_name_normalized", record)
        self.assertIn("business_name_normalized", result)


    def test_normalize_record_handles_missing_matching_columns(self):
        record = {
            "entity_id": "123",
            "country": "US",
        }

        result = normalize_record(record)

        self.assertEqual(result, record)


    def test_normalize_dataframe_adds_normalized_columns(self):
        

        dataframe = pd.DataFrame({
            "business_name": [
                "ACME, Inc.",
                "Orelee's Barbershop",
                "राम मार्केटिंग",
            ],
            "business_address": [
                "12 Main St.",
                "45 High Road",
                None,
            ],
        })

        result = normalize_dataframe(dataframe)

        self.assertEqual(
            result["business_name_normalized"].tolist(),
            [
                "acme inc",
                "orelee s barbershop",
                "राम मार्केटिंग",
            ],
        )

        self.assertEqual(
            result["business_address_normalized"].tolist(),
            [
                "12 main st",
                "45 high road",
                "",
            ],
        )


    def test_normalize_dataframe_does_not_modify_original(self):
        

        dataframe = pd.DataFrame({
            "business_name": ["ACME Inc."],
            "business_address": ["12 Main St."],
        })

        result = normalize_dataframe(dataframe)

        self.assertNotIn(
            "business_name_normalized",
            dataframe.columns,
        )

        self.assertIn(
            "business_name_normalized",
            result.columns,
        )


if __name__ == "__main__":
    unittest.main()
