# Set Transformer

## Main comparison: test top-1

Zero-filled, width 16, word, n = 1. Units: pooled units (union of all authors' top-k).

| k | signature | units | TF-SVM | Δ-sum | ST learned | ST background Δ |
|---:|---|---:|---:|---:|---:|---:|
| 1 | effect vector | 7 | 40.9 | 53.4 | 50.7 | 49.6 |
| 1 | relative frequency | 8 | 45.4 | 58.0 | 52.8 | 56.0 |
| 2 | effect vector | 7 | 40.9 | 62.3 | 50.7 | 49.6 |
| 2 | relative frequency | 12 | 51.2 | 61.7 | 57.9 | 60.0 |
| 5 | effect vector | 24 | 50.8 | 62.2 | 62.0 | 61.9 |
| 5 | relative frequency | 31 | 60.1 | 65.5 | 67.1 | 66.8 |
| 10 | effect vector | 57 | 59.4 | 64.0 | 68.1 | 70.4 |
| 10 | relative frequency | 58 | 64.9 | 66.0 | 71.6 | 70.1 |
| 20 | effect vector | 125 | 62.4 | 60.0 | 73.5 | 74.7 |
| 20 | relative frequency | 116 | 67.3 | 65.4 | 73.2 | 74.5 |
| 50 | effect vector | 316 | 67.4 | 60.8 | 75.1 | 75.5 |
| 50 | relative frequency | 284 | 70.6 | 63.8 | 74.7 | 76.4 |
| all | both (identical) | 2384 | 73.6 | 59.8 | 71.5 | 77.0 |

- k = all: both signatures select every cached unit, so their inputs and results are identical.
- Effect vector, k = 1 and 2: the same 7 pooled units, so ST and TF-SVM are identical; Δ-sum differs because it uses each author's own signature.

## Holdout top-1

Width 16; the train holdout (10 % per author) used for early stopping. Test was not used for any choice.

| k | effect vector, learned | effect vector, background Δ | relative frequency, learned | relative frequency, background Δ |
|---:|---:|---:|---:|---:|
| 1 | 54.7 | 52.8 | 53.8 | 57.2 |
| 2 | 54.7 | 52.8 | 58.8 | 63.0 |
| 5 | 66.4 | 65.8 | 66.7 | 69.6 |
| 10 | 72.0 | 73.9 | 74.5 | 73.5 |
| 20 | 75.8 | 74.8 | 76.2 | 77.0 |
| 50 | 78.0 | 78.0 | 78.0 | 78.7 |
| all | 75.7 | 78.6 | 75.7 | 78.6 |

## Width sweep (k = 10)

Holdout top-1 per width, zero-filled. Width 16 is best in 3 of the 4 effect vector / relative frequency rows and is used for every k.

| signature | unit vectors | 8 | 16 | 32 | 64 | 128 |
|---|---|---:|---:|---:|---:|---:|
| effect vector | learned | 68.6 | 72.0 | 67.2 | – | – |
| effect vector | background Δ | 72.3 | 73.9 | 72.4 | – | – |
| relative frequency | learned | 68.5 | 74.5 | 71.0 | 65.1 | 66.6 |
| relative frequency | background Δ | 69.5 | 73.5 | 74.9 | 73.2 | 34.9 |
| cosine of the means | learned | – | – | 69.9 | 65.1 | 63.2 |
| cosine of the means | background Δ | – | – | 72.7 | 71.0 | 68.9 |

Widths 64 and 128 and the cosine of the means rows are from the 11-13 run (older matching file, same scoring file).

Test top-1 of the width chosen on holdout in each run:

| signature | unit vectors | width 8 run | widths 16, 32 run | widths 32, 64, 128 run |
|---|---|---:|---:|---:|
| effect vector | learned | 68.3 | 68.1 (16) | – |
| effect vector | background Δ | 67.2 | 70.4 (16) | – |
| relative frequency | learned | 67.2 | 71.6 (16) | 67.8 (32) |
| relative frequency | background Δ | 66.9 | 71.1 (32) | 71.1 (32) |
| cosine of the means | learned | – | – | 68.0 (32) |
| cosine of the means | background Δ | – | – | 70.3 (32) |

## Available-case check (k = 10)

Test top-1, width 16.

| signature | TF-SVM | Δ-sum | ST learned | ST background Δ |
|---|---:|---:|---:|---:|
| effect vector | 58.3 | 63.8 | 68.4 | 68.4 |
| relative frequency | 62.7 | 64.9 | 68.7 | 68.3 |

## Reference: test top-1

| method | top-1 |
|---|---:|
| full LUAR (ceiling) | 88.1 |
| random, k = 10 (TF-SVM / Δ-sum, zero-filled) | 30.9 / 38.4 |

## Graph

Line chart of test top-1 over k (1, 2, 5, 10, 20, 50, all): TF-SVM, Δ-sum, ST learned and ST background Δ on effect vector and relative frequency, with the full-LUAR ceiling as a dashed line.

## Sources

Folder: results/effects/darkreddit/doc_mean/luar-mud_plain/word/n1/

| file | used for |
|---|---|
| set_transformer_20261116-170007.json | k = 1: main, holdout |
| set_transformer_20261118-183433.json | k = 2: main, holdout |
| set_transformer_20261116-153948.json | k = 5: main, holdout |
| set_transformer_20261118-184312.json | k = 10: main, holdout |
| set_transformer_20261116-154143.json | k = 20: main, holdout |
| set_transformer_20261116-165311.json | k = 50: main, holdout |
| set_transformer_20261118-181245.json | k = all: main, holdout |
| set_transformer_20261116-153129.json | width sweep: width 8 |
| set_transformer_20261116-152140.json | width sweep: widths 16, 32 |
| set_transformer_20261113-171100.json | width sweep: widths 32, 64, 128; cosine of the means |
| matching_20261116-151138.json | main: TF-SVM, Δ-sum; reference |
| available-case/set_transformer_20261116-153635.json | available-case check |
| available-case/matching_20261116-151045.json | available-case check |
