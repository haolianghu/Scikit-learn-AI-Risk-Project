"""Explore AI risk classification with matched pairs kept in the same fold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGET = "ProblematicOrProblemFreeAlgorithm"
GROUP = "PairID"


def prepare(original, cleaned):
    """Verify row alignment before using pair IDs from the original workbook."""
    if not original.columns.is_unique or not cleaned.columns.is_unique:
        raise ValueError("Workbook columns must be unique.")
    if GROUP not in original or TARGET not in cleaned:
        raise ValueError("PairID and the classification target are required.")
    missing = sorted(set(cleaned.columns) - set(original.columns))
    if missing:
        raise ValueError(f"Cleaned columns missing from original: {', '.join(missing)}")
    if len(original) != len(cleaned) or not original[cleaned.columns].reset_index(
        drop=True
    ).equals(cleaned.reset_index(drop=True)):
        raise ValueError("Original and cleaned rows do not match in the same order.")
    if original[GROUP].isna().any():
        raise ValueError("Pair IDs must not be missing.")
    labels = cleaned[TARGET]
    if labels.isna().any() or set(labels.unique()) != {0, 1}:
        raise ValueError("The target must contain both binary classes 0 and 1.")
    grouped = pd.DataFrame({GROUP: original[GROUP], TARGET: labels}).groupby(GROUP)[TARGET]
    if not ((grouped.size() == 2) & (grouped.nunique() == 2)).all():
        raise ValueError("Each pair must contain one example from each class.")

    # The dictionary dates these governance indicators to one year before
    # emergence; other cleaned columns may encode information learned later.
    features = []
    for number in range(100, 118):
        matching = [c for c in cleaned.columns if c.startswith(f"[{number}]")]
        if len(matching) != 1:
            raise ValueError(f"Expected exactly one governance indicator [{number}].")
        features.extend(matching)
    X = cleaned[features].apply(pd.to_numeric, errors="raise")
    if X.isna().any().any() or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Governance indicators must be finite and non-missing.")
    if not X.isin([0, 1, 2]).all().all():
        raise ValueError("Governance indicators must use the documented 0, 1, 2 coding.")
    return X, labels, original[GROUP]


def evaluate(original, cleaned, folds=5):
    X, y, groups = prepare(original, cleaned)
    if folds < 2 or folds > groups.nunique():
        raise ValueError("Folds must be between 2 and the number of matched pairs.")
    cv = GroupKFold(n_splits=folds)
    models = {
        "majority": DummyClassifier(strategy="prior"),
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
        ),
    }
    results = {}
    for name, model in models.items():
        probabilities = cross_val_predict(
            model, X, y, groups=groups, cv=cv, method="predict_proba"
        )[:, 1]
        predicted = (probabilities > 0.5).astype(int)
        results[name] = {
            "roc_auc": float(roc_auc_score(y, probabilities)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        }
    return {
        "rows": len(y),
        "pairs": int(groups.nunique()),
        "folds": folds,
        "features": list(X.columns),
        "metrics": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=Path("AI Data Original.xlsx"))
    parser.add_argument("--cleaned", type=Path, default=Path("AI Data Cleaned.xlsx"))
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()
    try:
        result = evaluate(pd.read_excel(args.original), pd.read_excel(args.cleaned), args.folds)
    except (OSError, ValueError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
