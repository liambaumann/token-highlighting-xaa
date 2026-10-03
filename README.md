# Token-Level Highlighting for Explainable Authorship Attribution
TU Wien Bachelor thesis repository

## Dependencies
```
pip install -r requirements.txt
```

## Usage
### Reuters
Requires Reuters_50_50 dataset extracted into:
```
data/reuter_50_50/C50train
data/reuter_50_50/C50test
```

### darkreddit
Requires the darkreddit authorship-attribution jsonl files extracted into:
```
data/darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_train_anon.jsonl
data/darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl
```

First run calculates centroids for every author and caches them under `cache/<dataset>_<model>/`

### Session problems with vsc.sh
Run `ssh -O exit vsc5` to close the old shared connection.
Run `ssh -fN vsc5` and log in.