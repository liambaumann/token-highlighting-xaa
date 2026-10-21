# reading and aggregating saved signature results; no model import, so signature.plot can
# use it without loading LUAR

import os
import json
from .keep import SETTINGS

RESULTS_ROOT = "results/signature"


def results_dir(dataset, perturbation, model_mode_name):
    return os.path.join(RESULTS_ROOT, dataset, perturbation, model_mode_name)


def load_runs(directory):
    # every run file in the folder, oldest first (file names are start timestamps)
    if not os.path.isdir(directory):
        return []
    runs = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith("keep_") and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                runs.append(json.load(f))
    return runs


def latest_series(runs):
    # (series id, its runs oldest first) for the series of the most recent run
    if not runs:
        return None, []
    series = runs[-1]["series"]
    return series, [run for run in runs if run["series"] == series]


def series_documents(series_runs):
    # all documents of a series, one entry per (author, doc_index)
    docs = {}
    for run in series_runs:
        for doc in run["documents"]:
            docs[(doc["author"], doc["doc_index"])] = doc
    return list(docs.values())


def metrics(entries):
    # entries: {"similarity", "rank"} per document - top-1 and MAP as in retrieval.py
    # (MAP there is the mean reciprocal rank of the true author)
    num = len(entries)
    return {
        "similarity": sum(e["similarity"] for e in entries) / num,
        "top1": sum(1 for e in entries if e["rank"] == 1) / num,
        "map": sum(1.0 / e["rank"] for e in entries) / num,
    }


def aggregate(documents, max_k):
    # mean over documents of the baseline and of every (setting, k); a document with k or fewer
    # content tokens counts with its unchanged (baseline) values at that k
    curves = {}
    for setting in SETTINGS:
        curves[setting] = []
        for i in range(max_k):
            entries = [doc["keep"][setting][i] for doc in documents]
            row = metrics(entries)
            row["k"] = entries[0]["k"]
            row["num_unchanged"] = sum(1 for e in entries if e["unchanged"])
            curves[setting].append(row)
    return {
        "num_documents": len(documents),
        "baseline": metrics([doc["baseline"] for doc in documents]),
        "curves": curves,
    }
