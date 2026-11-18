# Damping sensitivity of score 1

Date: 2026-10-22 18:14
Scoring file: scoring_20261022-180442.json (deltas: deltas_20261022-180442.pt, cache series 20261021-215418)
Matching file (check reference): matching_20261022-180448.json
Data: darkreddit, luar-mud, plain, doc_mean, units: subtoken; 2276 test comments, 6817 train comments, 10 authors.

What was done: score 1 final was recomputed from the saved score1_corrected as score1_corrected × n / (n + d), n = number of the author's sampled val comments containing the unit, for d in [5, 10, 20, 50]; each author's cached units were ranked by it (highest first, ties by unit string). Each author's top k units then went through both matching methods (TF-SVM and Δ-sum) with exactly the settings and code of effects.matching (fitting on val/train, evaluation on test). Nothing else was changed. Built-in check: d = 5 reproduces score 1 and d = 20 reproduces score 1a of the matching file exactly (same rankings and same top-1 for every method and k).

## TF-SVM: top-1 accuracy on test

| d | k = 1 | k = 2 | k = 5 | k = 10 | k = 20 | k = 50 | k = all |
|---|---|---|---|---|---|---|---|
| 5 | 41.70% | 44.64% | 49.78% | 56.15% | 61.29% | 66.96% | 73.55% |
| 10 | 42.18% | 45.87% | 53.69% | 59.01% | 62.92% | 67.88% | 73.55% |
| 20 | 41.74% | 48.24% | 56.02% | 60.63% | 64.15% | 68.37% | 73.55% |
| 50 | 41.74% | 47.45% | 57.95% | 60.72% | 65.03% | 68.76% | 73.55% |

## Δ-sum: top-1 accuracy on test

| d | k = 1 | k = 2 | k = 5 | k = 10 | k = 20 | k = 50 | k = all |
|---|---|---|---|---|---|---|---|
| 5 | 52.37% | 56.11% | 60.28% | 59.01% | 57.38% | 57.21% | 59.58% |
| 10 | 53.08% | 59.31% | 61.20% | 61.91% | 59.58% | 57.56% | 59.58% |
| 20 | 58.66% | 61.78% | 61.91% | 63.53% | 63.49% | 57.21% | 59.58% |
| 50 | 58.66% | 63.62% | 67.27% | 65.60% | 64.02% | 58.70% | 59.58% |

## Coverage

Share of test comments that contain at least one of their own author's top-k units.

| d | k = 1 | k = 5 | k = 10 |
|---|---|---|---|
| 5 | 59.7% | 77.9% | 92.0% |
| 10 | 68.1% | 90.8% | 94.9% |
| 20 | 76.9% | 91.0% | 96.2% |
| 50 | 76.9% | 95.1% | 97.7% |
