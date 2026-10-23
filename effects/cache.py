# effects, step 1: a cache of raw LUAR embeddings, from which later scoring and matching is
# computed locally. Builds and saves the cache only.
# Run from the repo root: python -u -m effects.cache   (--unit: unit type, default subtoken;
# --n: n-gram size, default 1)
#
# For every sampled SPLIT comment: emb(text), and for every distinct unit (n-gram for n >= 2) of the
# comment that occurs at least MIN_AUTHOR_COUNT times in its author's sampled comments, emb(text
# without it), all its positions filled at once per config.PERTURBATION (model.embed_variants, as
# scoring.occlusion_scores). After all of them, emb(text) of every test comment, same series.
# Results: results/effects/{dataset}/{perturbation}/{model_mode}/{unit}/n{n}/.
#
# Resumable as signature.scores: a run stops starting new comments after TIME_BUDGET_SECONDS,
# and the next run continues the same series with the comments not done yet. Every run writes
# its own timestamped files, embeddings (.pt) and metadata (.json, same name stem); nothing is
# overwritten.

import time
script_start = time.time()

import os
import json
import random
import argparse
from collections import Counter
from datetime import datetime
import torch
import config
from model import tokenizer, embed_variants, content_token_count
from data import DATASETS, load_authors, load_author_texts, load_jsonl_by_author, load_test_texts
from scoring import tokenize
from runs import results_dir, load_runs, latest_series, series_documents
from .units import UNIT_TYPES, NGRAM_SIZES, comment_units

if config.MODE != "plain":
    raise ValueError(f"effects.cache supports MODE \"plain\" only, not {config.MODE!r}.")

RESULTS_ROOT = "results/effects"
VAL_PREFIX = "val_"
TEST_PREFIX = "test_"

# sampling as in signature.scores: the split that gets sampled; the train split gives the
# centroids, the test split is only embedded whole
SPLIT = "val"
COMMENTS_PER_AUTHOR = 50
SAMPLE_SEED = 0
# signature.scores' eligibility rule (its n = 1)
MIN_CONTENT_TOKENS = 1
# a unit gets a removal embedding only if it occurs at least this often in its author's sampled comments
MIN_AUTHOR_COUNT = 2
# wall time from script start after which no new comment is started (devel jobs get 10 min)
TIME_BUDGET_SECONDS = 8 * 60
# seconds per (sequence token x embedded variant), as in signature.scores; replaced by this run's
# own measurement after the first comment
INITIAL_SECONDS_PER_TOKEN_VARIANT = 0.06e-3
# the files are rewritten at most this often (and at the end of each phase), not after every
# comment, since the .pt file grows to tens of MB
SAVE_INTERVAL_SECONDS = 30


def settings_info(unit, n):
    return {
        "split": SPLIT,
        "comments_per_author": COMMENTS_PER_AUTHOR,
        "sample_seed": SAMPLE_SEED,
        "min_content_tokens": MIN_CONTENT_TOKENS,
        "unit": unit,
        "n": n,
        "min_author_count": MIN_AUTHOR_COUNT,
    }


# load_split_texts and plan_comments: copied unchanged from signature/scores.py

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


def prepare_comment(text, unit, n=1):
    # tokenized as scoring.tokenize does (truncated to 512 incl. <s></s>); the comment's distinct
    # units (n-grams for n >= 2) -> all their positions, and its occurrences for the author counts
    input_ids, attention_mask = tokenize(text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    occurrences = comment_units(tokens, unit, n)
    positions = {}
    for name, unit_positions in occurrences:
        positions.setdefault(name, []).extend(unit_positions)
    # overlapping n-gram occurrences share positions: the union, in order (single units never overlap)
    positions = {name: sorted(set(unit_positions)) for name, unit_positions in positions.items()}
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokens": tokens,
        "n": n,
        "positions": positions,
        "occurrences": [name for name, _ in occurrences],
    }


def cached_units(prepared, counts):
    # the comment's distinct units that occur at least MIN_AUTHOR_COUNT times for its author,
    # in order of first position
    return [name for name in prepared["positions"] if counts[name] >= MIN_AUTHOR_COUNT]


def cache_comment(author, doc_index, prepared, counts, first_row):
    # emb(text) and emb(text without a) for every cached unit a, as (metadata, embeddings);
    # rows of the embeddings are numbered from first_row in the run's .pt file
    input_ids = prepared["input_ids"]
    num_tokens = input_ids.shape[1]
    units = cached_units(prepared, counts)

    # as in scoring.occlusion_scores: the base keeps every position, a unit's variant every
    # position except all of the unit's
    keep_lists = [list(range(num_tokens))]
    for name in units:
        removed = set(prepared["positions"][name])
        keep_lists.append([p for p in range(num_tokens) if p not in removed])
    embeddings = embed_variants(input_ids, prepared["attention_mask"], keep_lists)

    doc = {
        "author": author,
        "doc_index": doc_index,
        "token_ids": input_ids[0].tolist(),
        "tokens": prepared["tokens"],
        "base_row": first_row,
        "units": [
            {
                "unit": name,
                # n-grams: identity = the list of their units, with their n
                **({"ngram": prepared["n"]} if prepared["n"] > 1 else {}),
                "positions": prepared["positions"][name],
                "author_count": counts[name],
                "row": first_row + i,
            }
            for i, name in enumerate(units, start=1)
        ],
    }
    return doc, embeddings


def save_part(json_path, result, rows):
    # embeddings first, then the metadata that indexes them; each written to a temporary file and
    # moved into place, so an interrupted save leaves the previous consistent pair
    pt_path = os.path.join(os.path.dirname(json_path), result["embeddings_file"])
    torch.save(torch.cat(rows).to(torch.float32), pt_path + ".tmp")
    os.replace(pt_path + ".tmp", pt_path)
    with open(json_path + ".tmp", "w") as f:
        json.dump(result, f)
    os.replace(json_path + ".tmp", json_path)


def run_part(json_path, result, num_done_before, todo, estimate, compute, timing):
    # computes the todo comments in order until the time budget would be exceeded, appending to
    # result["documents"] and to the run's embedding rows; returns whether all of todo got done
    rows = []
    num_rows = 0
    last_save = time.time()
    finished = True
    for item in todo:
        work = estimate(item)
        elapsed = time.time() - script_start
        if elapsed + work * timing["seconds_per_unit"] > TIME_BUDGET_SECONDS:
            print(f"\ntime budget: {elapsed:.0f}s used, next comment estimated {work * timing['seconds_per_unit']:.0f}s - stopping, run again to continue")
            finished = False
            break

        doc_start = time.time()
        doc, embeddings = compute(item, num_rows)
        doc["seconds"] = time.time() - doc_start
        timing["seconds"] += doc["seconds"]
        timing["work"] += work
        timing["seconds_per_unit"] = timing["seconds"] / timing["work"]

        rows.append(embeddings)
        num_rows += embeddings.shape[0]
        result["documents"].append(doc)
        result["complete"] = num_done_before + len(result["documents"]) >= len(result["plan"])
        if time.time() - last_save >= SAVE_INTERVAL_SECONDS:
            save_part(json_path, result, rows)
            last_save = time.time()

    if result["documents"]:
        save_part(json_path, result, rows)
        print(f"saved {len(result['documents'])} comments, {num_rows} embeddings to {json_path}")
    return finished


def parse_args():
    parser = argparse.ArgumentParser(description="cache raw LUAR embeddings for the effects experiment")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="subtoken")
    parser.add_argument("--n", type=int, choices=NGRAM_SIZES, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    unit, n = args.unit, args.n
    started_at = datetime.now()
    stamp = f"{started_at:%Y%m%d-%H%M%S}"
    settings = settings_info(unit, n)
    print(f"dataset: {config.DATASET}, split: {SPLIT}, mode: {config.MODE}, model: {config.MODEL}, "
          f"perturbation: {config.PERTURBATION}, unit: {unit}, n: {n}")

    authors = load_authors(config.DATASET)
    plan, skipped = plan_comments(authors)
    texts_by_author = {author: load_split_texts(config.DATASET, SPLIT, author) for author in authors}
    prepared = {
        (author, doc_index): prepare_comment(texts_by_author[author][doc_index], unit, n)
        for author, doc_index in plan
    }
    # per author: occurrences of each unit over all of the author's sampled comments
    counts = {author: Counter() for author in authors}
    for (author, _), comment in prepared.items():
        counts[author].update(comment["occurrences"])

    test_texts = {author: load_test_texts(config.DATASET, author) for author in authors}
    test_plan = [[author, doc_index] for author in authors for doc_index in range(len(test_texts[author]))]

    out_dir = os.path.join(results_dir(RESULTS_ROOT, config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME), unit, f"n{n}")
    os.makedirs(out_dir, exist_ok=True)

    # continue the latest series if it has the same settings and plans and isn't complete yet
    series, val_runs = latest_series(load_runs(out_dir, VAL_PREFIX))
    test_runs = [run for run in load_runs(out_dir, TEST_PREFIX) if run["series"] == series]
    val_done = series_documents(val_runs)
    test_done = series_documents(test_runs)
    last = val_runs[-1] if val_runs else None
    if (last is None or last["settings"] != settings or last["plan"] != plan
            or (test_runs and test_runs[-1]["plan"] != test_plan)
            or (len(val_done) >= len(plan) and len(test_done) >= len(test_plan))):
        series = stamp
        val_runs, test_runs, val_done, test_done = [], [], [], []
        print(f"starting new series {series}")
    else:
        print(f"continuing series {series}: {len(val_done)}/{len(plan)} {SPLIT} comments, "
              f"{len(test_done)}/{len(test_plan)} test comments done")

    common = {
        "series": series,
        "started_at": started_at.isoformat(timespec="seconds"),
        "dataset": config.DATASET,
        "model": config.MODEL,
        "mode": config.MODE,
        "perturbation": config.PERTURBATION,
        "authors": authors,
    }
    timing = {"seconds": 0.0, "work": 0, "seconds_per_unit": INITIAL_SECONDS_PER_TOKEN_VARIANT}

    # sampled SPLIT comments, each with its cached units removed
    done = {(doc["author"], doc["doc_index"]) for doc in val_done}
    val_result = {
        "experiment": "effects_cache",
        "part": len(val_runs) + 1,
        **common,
        "settings": settings,
        "plan": plan,
        "skipped_short_documents": skipped,
        "complete": False,
        "embeddings_file": f"{VAL_PREFIX}{stamp}.pt",
        "documents": [],
    }

    def estimate_val(item):
        comment = prepared[tuple(item)]
        return comment["input_ids"].shape[1] * (1 + len(cached_units(comment, counts[item[0]])))

    def compute_val(item, first_row):
        author, doc_index = item
        doc, embeddings = cache_comment(author, doc_index, prepared[(author, doc_index)], counts[author], first_row)
        print(f"{author}, comment {doc_index}, {len(doc['token_ids']) - 2} tokens, {len(doc['units'])} units")
        return doc, embeddings

    val_finished = run_part(
        os.path.join(out_dir, f"{VAL_PREFIX}{stamp}.json"), val_result, len(val_done),
        [item for item in plan if tuple(item) not in done], estimate_val, compute_val, timing,
    )
    num_val = len(val_done) + len(val_result["documents"])

    # every test comment, whole, once all SPLIT comments of the series are done
    num_test = len(test_done)
    if val_finished:
        done = {(doc["author"], doc["doc_index"]) for doc in test_done}
        test_result = {
            "experiment": "effects_cache_test",
            "part": len(test_runs) + 1,
            **common,
            "plan": test_plan,
            "complete": False,
            "embeddings_file": f"{TEST_PREFIX}{stamp}.pt",
            "documents": [],
        }

        def estimate_test(item):
            return content_token_count(test_texts[item[0]][item[1]]) + 2

        def compute_test(item, first_row):
            author, doc_index = item
            input_ids, attention_mask = tokenize(test_texts[author][doc_index])
            embedding = embed_variants(input_ids, attention_mask, [list(range(input_ids.shape[1]))])
            return {"author": author, "doc_index": doc_index, "row": first_row}, embedding

        run_part(
            os.path.join(out_dir, f"{TEST_PREFIX}{stamp}.json"), test_result, len(test_done),
            [item for item in test_plan if tuple(item) not in done], estimate_test, compute_test, timing,
        )
        num_test += len(test_result["documents"])

    complete = num_val >= len(plan) and num_test >= len(test_plan)
    print(f"\nseries {series}: {num_val}/{len(plan)} {SPLIT} comments, {num_test}/{len(test_plan)} test comments done"
          + ("" if complete else " - run again to continue"))


if __name__ == "__main__":
    main()
