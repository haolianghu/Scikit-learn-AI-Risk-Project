import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from risk_baseline import GROUP, TARGET, evaluate, prepare


ROOT = Path(__file__).resolve().parents[1]


class RiskBaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = pd.read_excel(ROOT / "AI Data Original.xlsx")
        cls.cleaned = pd.read_excel(ROOT / "AI Data Cleaned.xlsx")

    def test_existing_workbooks(self):
        result = evaluate(self.original, self.cleaned)
        self.assertEqual((result["rows"], result["pairs"]), (94, 47))
        self.assertEqual(len(result["features"]), 18)
        for metrics in result["metrics"].values():
            self.assertTrue(np.isfinite(metrics["roc_auc"]))
            self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), 94)

    def test_pairs_never_cross_folds(self):
        X, y, groups = prepare(self.original, self.cleaned)
        for train, test in GroupKFold(n_splits=5).split(X, y, groups):
            self.assertFalse(set(groups.iloc[train]) & set(groups.iloc[test]))

    def test_misaligned_workbooks_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "do not match"):
            prepare(self.original, self.cleaned.iloc[::-1].reset_index(drop=True))

    def test_invalid_pair_is_rejected(self):
        original = self.original.copy()
        original.loc[0, GROUP] = -1
        with self.assertRaisesRegex(ValueError, "Each pair"):
            prepare(original, self.cleaned)

    def test_missing_values_are_rejected(self):
        original = self.original.copy()
        cleaned = self.cleaned.copy()
        feature = cleaned.columns[0]
        original.loc[0, feature] = np.nan
        cleaned.loc[0, feature] = np.nan
        with self.assertRaisesRegex(ValueError, "finite and non-missing"):
            prepare(original, cleaned)

    def test_target_is_not_a_predictor(self):
        X, _, _ = prepare(self.original, self.cleaned)
        self.assertNotIn(TARGET, X.columns)
        self.assertNotIn(GROUP, X.columns)


if __name__ == "__main__":
    unittest.main()
