# Set Transformer

## Runs

### k = 10, zero-filled, widths 32, 64, 128

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| cosine distance | learned | 32 | 69.9 | 68.0 | 57.5 | 60.8 |
| cosine distance | background Δ | 32 | 72.7 | 70.3 | 57.5 | 60.8 |
| relative frequency | learned | 32 | 71.0 | 67.8 | 64.9 | 66.0 |
| relative frequency | background Δ | 32 | 74.9 | 71.1 | 64.9 | 66.0 |

### k = 10, zero-filled, widths 16, 32

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| Euclidean distance | learned | 16 | 72.0 | 68.1 | 59.4 | 64.0 |
| Euclidean distance | background Δ | 16 | 73.9 | 70.4 | 59.4 | 64.0 |
| relative frequency | learned | 16 | 74.5 | 71.6 | 64.9 | 66.0 |
| relative frequency | background Δ | 32 | 74.9 | 71.1 | 64.9 | 66.0 |

### k = 10, zero-filled, widths 8

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| Euclidean distance | learned | 8 | 68.6 | 68.3 | 59.4 | 64.0 |
| Euclidean distance | background Δ | 8 | 72.3 | 67.2 | 59.4 | 64.0 |
| relative frequency | learned | 8 | 68.5 | 67.2 | 64.9 | 66.0 |
| relative frequency | background Δ | 8 | 69.5 | 66.9 | 64.9 | 66.0 |

### k = 5, zero-filled, widths 16

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| Euclidean distance | learned | 16 | 66.4 | 62.0 | 50.8 | 62.2 |
| Euclidean distance | background Δ | 16 | 65.8 | 61.9 | 50.8 | 62.2 |
| relative frequency | learned | 16 | 66.7 | 67.1 | 60.1 | 65.5 |
| relative frequency | background Δ | 16 | 69.6 | 66.8 | 60.1 | 65.5 |

### k = 20, zero-filled, widths 16

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| Euclidean distance | learned | 16 | 75.8 | 73.5 | 62.4 | 60.0 |
| Euclidean distance | background Δ | 16 | 74.8 | 74.7 | 62.4 | 60.0 |
| relative frequency | learned | 16 | 76.2 | 73.2 | 67.3 | 65.4 |
| relative frequency | background Δ | 16 | 77.0 | 74.5 | 67.3 | 65.4 |

### k = 10, available-case, widths 16

| signature | unit vectors | chosen width | holdout top-1 | test top-1 | TF-SVM | Δ-sum |
|---|---|---:|---:|---:|---:|---:|
| Euclidean distance | learned | 16 | 69.1 | 68.4 | 58.3 | 63.8 |
| Euclidean distance | background Δ | 16 | 71.1 | 68.4 | 58.3 | 63.8 |
| relative frequency | learned | 16 | 69.9 | 68.7 | 62.7 | 64.9 |
| relative frequency | background Δ | 16 | 69.9 | 68.3 | 62.7 | 64.9 |

## Main comparison: test top-1

| mean | k | signature | TF-SVM | Δ-sum | ST learned | ST background Δ |
|---|---:|---|---:|---:|---:|---:|
| zero-filled | 5 | Euclidean distance | 50.8 | 62.2 | 62.0 | 61.9 |
| zero-filled | 5 | relative frequency | 60.1 | 65.5 | 67.1 | 66.8 |
| zero-filled | 10 | Euclidean distance | 59.4 | 64.0 | 68.1 | 70.4 |
| zero-filled | 10 | relative frequency | 64.9 | 66.0 | 71.6 | 71.1 |
| zero-filled | 20 | Euclidean distance | 62.4 | 60.0 | 73.5 | 74.7 |
| zero-filled | 20 | relative frequency | 67.3 | 65.4 | 73.2 | 74.5 |
| available-case | 10 | Euclidean distance | 58.3 | 63.8 | 68.4 | 68.4 |
| available-case | 10 | relative frequency | 62.7 | 64.9 | 68.7 | 68.3 |

## Reference: test top-1

| method | top-1 |
|---|---:|
| full LUAR (ceiling) | 88.1 |
| TF-SVM, all units | 73.6 |
| Δ-sum, all units (zero-filled) | 59.8 |
| random, k = 10 (TF-SVM / Δ-sum, zero-filled) | 30.9 / 38.4 |

## Graph

Line chart of test top-1 over k (5, 10, 20): TF-SVM, Δ-sum and ST learned on Euclidean distance and relative frequency, with the full-LUAR ceiling as a dashed line.

## Sources

Folder: results/effects/darkreddit/doc_mean/luar-mud_plain/word/n1/
- set_transformer_20261113-171100.json
- set_transformer_20261116-152140.json
- set_transformer_20261116-153129.json
- set_transformer_20261116-153948.json
- set_transformer_20261116-154143.json
- available-case/set_transformer_20261116-153635.json
- matching_20261116-151138.json
- available-case/matching_20261116-151045.json
