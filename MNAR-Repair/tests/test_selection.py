import unittest

from mnar_repair import greedy_repair_set, mnar_edges


class GreedyRepairSetTests(unittest.TestCase):
    def setUp(self):
        self.mgraph = {
            "education": {"mechanism": "MCAR", "parents": []},
            "tax": {"mechanism": "MNAR", "parents": ["education"]},
            "loan": {"mechanism": "MNAR", "parents": ["income"]},
            "income": {"mechanism": "MNAR", "parents": ["loan"]},
            "job": {"mechanism": "FullyObserved", "parents": []},
        }
        self.rates = {
            "education": 0.1,
            "tax": 0.2,
            "loan": 0.3,
            "income": 0.2,
        }

    def test_mnar_edges_require_incomplete_parent_and_child(self):
        self.assertEqual(
            mnar_edges(self.mgraph),
            {("education", "tax"), ("income", "loan"), ("loan", "income")},
        )

    def test_selection_counts_both_endpoints_of_an_edge(self):
        selected = greedy_repair_set(self.mgraph, missingness_rates=self.rates)
        self.assertEqual(selected, ["income", "education"])

    def test_cost_changes_the_selected_endpoint(self):
        costs = {"education": 1, "tax": 1, "loan": 1, "income": 100}
        selected = greedy_repair_set(
            self.mgraph,
            costs=costs,
            missingness_rates=self.rates,
        )
        self.assertEqual(selected, ["loan", "education"])

    def test_nonpositive_cost_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            greedy_repair_set(self.mgraph, costs={"income": 0})


if __name__ == "__main__":
    unittest.main()
