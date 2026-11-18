# Frequency signature: no damping vs d = 5

Date: 2026-10-22 18:57
Matching file: matching_20261022-185317.json (scoring: scoring_20261022-185312.json, deltas: deltas_20261022-185312.pt, cache series 20261021-215418)
Data: darkreddit, luar-mud, plain, doc_mean, units: subtoken; 2276 test comments, 6817 train comments, 10 authors.

What was done: the frequency signature of effects.matching (freq = occurrences of the unit in the author's sampled val comments / all units in them, corrected = freq − mean over all authors of freq) was ranked twice: without damping (final = corrected) and with d = 5 (final = corrected × n / (n + 5), n = number of the author's comments containing the unit), highest first, ties by unit string. Each author's top k units went through both matching methods (TF-SVM and Δ-sum) with exactly the settings and code of effects.matching (fitting on val/train, evaluation on test). Built-in check: d = 5 reproduces the matching file's frequency ranking and its frequency top-1 for every method and k.

## TF-SVM: top-1 accuracy on test

| | k = 1 | k = 2 | k = 5 | k = 10 | k = 20 | k = 50 | k = all |
|---|---|---|---|---|---|---|---|
| no damping | 45.87% | 49.91% | 59.09% | 64.41% | 66.74% | 70.65% | 73.55% |
| d = 5 | 45.87% | 49.12% | 59.36% | 63.80% | 66.56% | 70.87% | 73.55% |

## Δ-sum: top-1 accuracy on test

| | k = 1 | k = 2 | k = 5 | k = 10 | k = 20 | k = 50 | k = all |
|---|---|---|---|---|---|---|---|
| no damping | 57.56% | 61.91% | 61.29% | 65.16% | 65.03% | 63.49% | 59.58% |
| d = 5 | 57.56% | 63.09% | 61.86% | 67.09% | 65.16% | 63.62% | 59.58% |

## Units in the top 10 without damping but not with d = 5

Per author; occurrences = total occurrences of the unit in the author's sampled val comments, comments = number of those comments containing it (n).

| author | unit | occurrences | comments |
|---|---|---|---|
| user1 | `Ġcould` | 32 | 18 |
| user10 | (none) | | |
| user2 | `Ġthread` | 24 | 14 |
| user3 | `'re` | 18 | 9 |
| user3 | `Ġfrom` | 23 | 17 |
| user4 | `Ġmeth` | 27 | 11 |
| user5 | `NM` | 16 | 11 |
| user6 | `Ġour` | 20 | 8 |
| user7 | `http` | 19 | 9 |
| user8 | `Ġthough` | 16 | 13 |
| user9 | `Ġor` | 44 | 12 |
