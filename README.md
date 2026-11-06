# Exploring organizational AI risk indicators

This project explores characteristics of problematic and problem-free
algorithms. The original notebook, presentation and executive summary document
the analysis. `AI Data Original.xlsx` contains the 94 examples, matched-pair
identifiers, and a data dictionary; `AI Data Cleaned.xlsx` contains the
predictors used in the original analysis and its own dictionary.

## Pair-aware baseline

`risk_baseline.py` compares a majority-class baseline with logistic regression
using out-of-fold predictions from five-fold grouped cross-validation.
Matched problematic and problem-free examples share a fold rather than
appearing on opposite sides of a split. The script verifies that the cleaned
workbook matches the original row by row before taking pair IDs from the
original, and checks that every pair contains one example from each class.

The model uses only the 18 governance indicators numbered 100–117, described
in the dictionary as measured one year before problem emergence. Scaling is
fitted inside each training fold. Output includes ROC AUC, balanced accuracy
and a confusion matrix (rows are actual 0/1, columns predicted 0/1) using a
0.5 probability threshold. A score of exactly 0.5 is classified as 0.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python risk_baseline.py
python -m unittest discover -s tests
```

Use `--original`, `--cleaned`, or `--folds` to change the input paths or
number of folds. Inputs are read from their first sheets and are never modified.
Workbooks must have matching cleaned columns in the same order; reordering
either independently is rejected.

## Limitations

This is a small, retrospectively matched sample, not a representative
deployment dataset. A matched-pair split does not prevent the same developer
from appearing in multiple folds; the reported scores do not establish
generalization to new organizations or future incidents. Data provenance,
label ascertainment and the availability of each indicator at prediction time
need independent verification. Observational associations and classifier
coefficients are not causal evidence. Do not use this baseline for operational
risk decisions without independent validation.

The original notebook uses a different set of predictors and a row-level split;
its feature selection and test-set cross-validation should not be interpreted
as an independent evaluation of a deployed model. Running that notebook
requires additional dependencies beyond the baseline requirements.
