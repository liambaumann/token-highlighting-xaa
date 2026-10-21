# reading saved series results (every run writes a timestamped file, runs continuing a
# series share its id); no model import, so report scripts can use it without loading LUAR

import os
import json


def results_dir(results_root, dataset, perturbation, model_mode_name):
    return os.path.join(results_root, dataset, perturbation, model_mode_name)


def load_runs(directory, prefix):
    # every run file in the folder whose name starts with prefix, oldest first (file names are
    # start timestamps); each run gets its file name under "file"
    if not os.path.isdir(directory):
        return []
    runs = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(prefix) and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                run = json.load(f)
            run["file"] = fname
            runs.append(run)
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
