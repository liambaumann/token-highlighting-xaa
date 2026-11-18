# McNemar test: per-comment results and first comparison

Date: 2026-11-16
Data: darkreddit, luar-mud, plain, doc_mean, units: word, n = 1. 2276 test comments.

## Code changes (not committed)

- `effects/matching.py`: every result entry (including each random seed) now also stores `correct_by_comment`, a 0/1 list in test-plan order. Its sum equals the existing `correct` count. Accuracy fields are unchanged. Matching files grow from about 1.5 MB to 6.5 MB.
- `effects/mcnemar.py` (new): paired test between two matching results.

```
python -m effects.mcnemar --side1 FILE SIG METHOD K --side2 FILE SIG METHOD K
python -m effects.mcnemar --side1 FILE SIG METHOD --side2 FILE SIG METHOD --all-k
```

- b = comments side 1 gets right and side 2 gets wrong, c = the opposite.
- p = `scipy.stats.binomtest(b, b + c)`, p = 1 when b + c = 0.
- Refuses files with different cache series, so only comparisons within the same unit type and n are possible.
- Refuses files without `correct_by_comment` (all files written before 2026-11-16).

## New result files

- zero-filled: `results/effects/darkreddit/doc_mean/luar-mud_plain/word/n1/matching_20261116-151138.json`
- available-case: `results/effects/darkreddit/doc_mean/luar-mud_plain/word/n1/available-case/matching_20261116-151045.json`

Accuracies are identical to the earlier runs.

## First comparison: available-case (1) vs zero-filled (2), TF-SVM, score 1

| k | acc 1 | acc 2 | b | c | p |
|---|---:|---:|---:|---:|---:|
| 1 | 40.82% | 33.48% | 303 | 136 | 1.05e-15 |
| 2 | 49.03% | 42.31% | 291 | 138 | 1.19e-13 |
| 5 | 54.96% | 55.05% | 112 | 114 | 0.947 |
| 10 | 60.76% | 57.47% | 183 | 108 | 1.3e-05 |
| 20 | 64.19% | 62.79% | 112 | 80 | 0.025 |
| 50 | 67.22% | 66.78% | 99 | 89 | 0.512 |
| all | 73.55% | 73.55% | 0 | 0 | 1 |

The available-case mean is significantly better for score 1 with TF-SVM at k = 1, 2 and 10 (p < 0.001) and k = 20 (p < 0.05). There is no difference at k = 5 and 50.

## Still to run

The comparison the thesis claim rests on: available-case, Δ-sum, score 2 vs frequency, `--all-k`.