import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mnar_repair import mnar_edges, repair_relation


DATA_DIRECTORY = Path(__file__).parent / "data"


class RepairPipelineTests(unittest.TestCase):
    def setUp(self):
        self.relation = pd.read_csv(DATA_DIRECTORY / "example_relation.csv")
        with (DATA_DIRECTORY / "example_mgraph.json").open(encoding="utf-8") as handle:
            self.mgraph = json.load(handle)

    def test_pipeline_repairs_only_selected_attributes(self):
        original = self.relation.copy(deep=True)
        result = repair_relation(self.relation, self.mgraph, imputer="simple")

        self.assertEqual(result.repair_set, ("income", "education"))
        self.assertFalse(result.relation["income"].isna().any())
        self.assertFalse(result.relation["education"].isna().any())
        self.assertTrue(result.relation["loan"].isna().any())
        self.assertTrue(result.relation["tax"].isna().any())
        self.assertEqual(mnar_edges(result.mgraph), set())

        for attribute in result.repair_set:
            observed = original[attribute].notna()
            pd.testing.assert_series_equal(
                result.relation.loc[observed, attribute],
                original.loc[observed, attribute],
            )

        pd.testing.assert_frame_equal(self.relation, original)
        self.assertEqual(self.mgraph["income"]["mechanism"], "MNAR")

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is not installed")
    def test_mbi_repairs_the_selected_attributes(self):
        result = repair_relation(self.relation, self.mgraph, imputer="mbi")
        self.assertEqual(result.repair_set, ("income", "education"))
        self.assertFalse(result.relation["income"].isna().any())
        self.assertFalse(result.relation["education"].isna().any())
        self.assertEqual(mnar_edges(result.mgraph), set())

    def test_cli_writes_outputs_and_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "output"
            command = [
                sys.executable,
                "-m",
                "mnar_repair",
                str(DATA_DIRECTORY / "example_relation.csv"),
                str(DATA_DIRECTORY / "example_mgraph.json"),
                "--costs",
                str(DATA_DIRECTORY / "example_costs.json"),
                "--imputer",
                "simple",
                "--output-dir",
                str(output_directory),
            ]
            first = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertTrue((output_directory / "repaired_relation.csv").is_file())
            self.assertTrue((output_directory / "repaired_mgraph.json").is_file())
            self.assertTrue((output_directory / "repair_set.json").is_file())

            second = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("Refusing to replace", second.stderr)


if __name__ == "__main__":
    unittest.main()
