retrieval.py results, 2026-09-30

reuters dataset, luar-mud model, all 50 authors, 2500 test docs.

USE_CHUNKS = False (embed whole doc up to 512 tokens, as before):
top-1 accuracy: 1281/2500 (~51.24%)
soft top-5 accuracy: 2230/2500 (~89.20%)
soft top-10 accuracy: 2425/2500 (~97.00%)
mean average precision: 67.33%

USE_CHUNKS = True (embed_chunked: 30-token pieces + <s>/</s>, padded to 32, all pieces
from a doc as one episode, first 510 tokens of doc only):
top-1 accuracy: 1405/2500 (~56.20%)
soft top-5 accuracy: 2225/2500 (~89.00%)
soft top-10 accuracy: 2426/2500 (~97.04%)
mean average precision: 70.39%

chunking gives a clear bump in top-1 (51.24% -> 56.20%) and MAP (67.33% -> 70.39%),
soft top-5/top-10 basically unchanged.

single32 mode:

mode                              | top-1          | soft top-5     | soft top-10    | MAP
plain                              | 51.24%         | 89.20%         | 97.00%         | 67.33%
chunk32                            | 56.20%         | 89.00%         | 97.04%         | 70.39%
single32 (mean ± std, seeds [0, 1, 2]) | 21.36% ± 0.14% | 54.29% ± 1.23% | 72.00% ± 0.35% | 36.92% ± 0.14%

## Results for DarkReddit
mode	top-1	soft top-5	soft top-10	MAP
plain	2006/2276 (88.14%)	2260/2276 (99.30%)	2276/2276 (100%)	93.03%
chunk32	1883/2276 (82.73%)	2218/2276 (97.45%)	2276/2276 (100%)	89.15%





but these results are not directly comparable, because darkreddit dataset is a 10-way classification problem, while reuters is 50-way. Further, note that only 50 of 500 docs are used in training to compute centroids.

For comparison, the 10-way classification results for reuters are:
plain:
top-1 accuracy: 409/500 (~81.80%)
soft top-5 accuracy: 500/500 (~100.00%)
soft top-10 accuracy: 500/500 (~100.00%)
mean average precision: 89.92%

chunk32:
top-1 accuracy: 431/500 (~86.20%)
soft top-5 accuracy: 500/500 (~100.00%)
soft top-10 accuracy: 500/500 (~100.00%)
mean average precision: 92.40%

acually better performance on soft top-5 and comparable results for the other metrics.