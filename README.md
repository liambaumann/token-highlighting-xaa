# Token-Level Highlighting for Explainable Authorship Attribution
TU Wien Bachelor thesis repository

## Dependencies
- torch
- transformers
- einops

## Usage
### Reuters
Requires Reuters_50_50 dataset extracted into:
```
data/C50train
data/C50test
```

First run calculates centroids for every author and caches them under `centroids.pt`