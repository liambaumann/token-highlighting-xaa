# signature scores: occlusion scores (n = 1) of a random sample of SPLIT comments per author,
# each scored against that author's centroid (from load_train_texts, cached by get_centroids).
# signature.tokens builds the per-author token signatures from these files, without the model.
# Run from the repo root: python -u -m signature.scores
#
# Resumable as signature.run: a run stops starting new comments after TIME_BUDGET_SECONDS, and
# the next run continues the same series with the comments not done yet, so one series can span
# several 10-minute devel jobs. Every run writes its own timestamped file, nothing is overwritten.

import time
script_start = time.time()

import os
import json
import random
from datetime import datetime
import config
from model import tokenizer, embed_chunked, content_token_count
from data import DATASETS, load_authors, load_author_texts, load_jsonl_by_author, get_centroids
from scoring import tokenize, occlusion_scores
from .results import results_dir, latest_series, series_documents
from .tokens import SCORES_PREFIX, load_score_runs

if config.MODE not in ("plain", "chunk32"):
    raise ValueError(f"signature.scores supports MODE \"plain\" and \"chunk32\", not {config.MODE!r}.")

# the split that gets scored; the train split gives the centroids, keep the final test split out
SPLIT = "val"
COMMENTS_PER_AUTHOR = 50
SAMPLE_SEED = 0
N = 1
# occlusion_scores needs at least n content tokens for a window
MIN_CONTENT_TOKENS = N
# wall time from script start after which no new comment is started (devel jobs get 10 min)
TIME_BUDGET_SECONDS = 8 * 60
# seconds per (sequence token x embedded variant), as in signature.run; replaced by this run's
# own measurement after the first comment
INITIAL_SECONDS_PER_TOKEN_VARIANT = 0.06e-3

SETTINGS_INFO = {
    "split": SPLIT,
    "comments_per_author": COMMENTS_PER_AUTHOR,
    "sample_seed": SAMPLE_SEED,
    "n": N,
    "min_content_tokens": MIN_CONTENT_TOKENS,
}


def load_split_texts(dataset, split, author):
    # all texts of one author in a split, in file order (doc_index is the index into this list)
    source = DATASETS[dataset]
    if split not in source:
        raise ValueError(f"dataset {dataset!r} has no split {split!r}, only {[k for k in source if k != 'type']}")
    if source["type"] == "folders":
        return load_author_texts(os.path.join(source[split], author))
    return load_jsonl_by_author(source[split])[author]


def plan_comments(authors):
    # COMMENTS_PER_AUTHOR random comments per author (one rng per author, seeded with SAMPLE_SEED),
    # drawn from the comments with at least MIN_CONTENT_TOKENS content tokens, in index order;
    # interleaved over authors (every author's first, then every author's second, ...) so a
    # series that is cut short is still balanced
    per_author = {}
    skipped = []
    for author in authors:
        texts = load_split_texts(config.DATASET, SPLIT, author)
        eligible = []
        for doc_index, text in enumerate(texts):
            if content_token_count(text) < MIN_CONTENT_TOKENS:
                skipped.append([author, doc_index])
            else:
                eligible.append(doc_index)
        if len(eligible) < COMMENTS_PER_AUTHOR:
            raise ValueError(f"{author} has only {len(eligible)} usable {SPLIT} comments, needs {COMMENTS_PER_AUTHOR}")
        rng = random.Random(SAMPLE_SEED)
        per_author[author] = sorted(rng.sample(eligible, COMMENTS_PER_AUTHOR))

    plan = [
        [author, per_author[author][i]]
        for i in range(COMMENTS_PER_AUTHOR)
        for author in authors
    ]
    return plan, skipped


def estimated_work(num_tokens):
    # embedded variants (base + one per window) x sequence length
    return num_tokens * (num_tokens - N)


def score_comment(author, doc_index, text, centroid):
    input_ids, attention_mask = tokenize(text)
    base_similarity, scores = occlusion_scores(input_ids, attention_mask, centroid, N)
    token_ids = input_ids[0].tolist()
    return {
        "author": author,
        "doc_index": doc_index,
        # content tokens the model sees (after truncation to 512 incl. <s></s>), and before truncation
        "num_content_tokens": len(token_ids) - 2,
        "full_content_tokens": content_token_count(text, max_tokens=10**9),
        "base_similarity": base_similarity,
        # aligned by position; <s> (0) and </s> (last) are never occluded and keep score 0.0
        "token_ids": token_ids,
        "tokens": tokenizer.convert_ids_to_tokens(token_ids),
        "scores": scores,
    }


def main():
    started_at = datetime.now()
    print(f"dataset: {config.DATASET}, split: {SPLIT}, mode: {config.MODE}, model: {config.MODEL}, "
          f"perturbation: {config.PERTURBATION}, n: {N}")

    authors = load_authors(config.DATASET)
    if config.MODE == "chunk32":
        centroids = get_centroids(config.DATASET, authors, embed_fn=embed_chunked, cache_name="centroids_chunk32.pt")
    else:
        centroids = get_centroids(config.DATASET, authors)

    plan, skipped = plan_comments(authors)
    out_dir = results_dir(config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # continue the latest series if it has the same settings and plan and isn't complete yet
    series, series_runs = latest_series(load_score_runs(out_dir))
    done_docs = series_documents(series_runs)
    last = series_runs[-1] if series_runs else None
    if last is None or last["settings"] != SETTINGS_INFO or last["plan"] != plan or len(done_docs) >= len(plan):
        series = f"{started_at:%Y%m%d-%H%M%S}"
        series_runs = []
        done_docs = []
        print(f"starting new series {series}")
    else:
        print(f"continuing series {series}: {len(done_docs)}/{len(plan)} comments done")

    done = {(doc["author"], doc["doc_index"]) for doc in done_docs}
    todo = [(author, doc_index) for author, doc_index in plan if (author, doc_index) not in done]

    result_path = os.path.join(out_dir, f"{SCORES_PREFIX}{started_at:%Y%m%d-%H%M%S}.json")
    result = {
        "experiment": "signature_scores",
        "series": series,
        "part": len(series_runs) + 1,
        "started_at": started_at.isoformat(timespec="seconds"),
        "dataset": config.DATASET,
        "model": config.MODEL,
        "mode": config.MODE,
        "perturbation": config.PERTURBATION,
        "authors": authors,
        "settings": SETTINGS_INFO,
        "plan": plan,
        "skipped_short_documents": skipped,
        "complete": False,
        "documents": [],
    }

    texts_by_author = {author: load_split_texts(config.DATASET, SPLIT, author) for author in authors}
    seconds_per_unit = INITIAL_SECONDS_PER_TOKEN_VARIANT
    total_seconds = 0.0
    total_work = 0
    for author, doc_index in todo:
        text = texts_by_author[author][doc_index]
        work = estimated_work(content_token_count(text) + 2)
        elapsed = time.time() - script_start
        if elapsed + work * seconds_per_unit > TIME_BUDGET_SECONDS:
            print(f"\ntime budget: {elapsed:.0f}s used, next comment estimated {work * seconds_per_unit:.0f}s - stopping, run again to continue")
            break

        doc_start = time.time()
        doc = score_comment(author, doc_index, text, centroids[author])
        doc["seconds"] = time.time() - doc_start
        total_seconds += doc["seconds"]
        total_work += work
        seconds_per_unit = total_seconds / total_work

        print(f"{author}, comment {doc_index}, {doc['num_content_tokens']} tokens: {doc['seconds']:.1f}s, "
              f"base similarity {doc['base_similarity']:.4f}")

        # saved after every comment so a job killed at its time limit loses at most one
        result["documents"].append(doc)
        result["complete"] = len(done_docs) + len(result["documents"]) >= len(plan)
        with open(result_path, "w") as f:
            json.dump(result, f)

    num_done = len(done_docs) + len(result["documents"])
    print(f"\nseries {series}: {num_done}/{len(plan)} comments done"
          + ("" if num_done >= len(plan) else " - run again to continue"))
    if result["documents"]:
        print(f"saved results to {result_path}")


if __name__ == "__main__":
    main()
