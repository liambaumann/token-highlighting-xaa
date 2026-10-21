# signature experiment: keep only the k highest-scoring tokens / n-grams of each test document
# (by occlusion score), fill everything else per config.PERTURBATION, and measure similarity to
# the author centroid and author retrieval (top-1, MAP) for k = 1..MAX_K, with n = 1, 2, 3 and
# the combined 1-3 ranking. Baseline is the full document.
# Run from the repo root: python -u -m signature.run
#
# Resumable: a run stops starting new documents after TIME_BUDGET_SECONDS, and the next run
# continues the same series with the documents not done yet, so one series can span several
# 10-minute devel jobs. Every run writes its own timestamped file, nothing is overwritten.

import time
script_start = time.time()

import os
import json
from datetime import datetime
import torch
import config
from model import tokenizer, embed_chunked, embed_variants, content_token_count
from data import load_authors, load_test_texts, get_centroids
from scoring import tokenize, occlusion_scores
from .keep import NGRAM_SIZES, COMBINED, SETTINGS, window_ranking, keep_top_windows, rank_average_order
from .results import results_dir, load_runs, latest_series, series_documents, aggregate

if config.MODE not in ("plain", "chunk32"):
    raise ValueError(f"signature.run supports MODE \"plain\" and \"chunk32\", not {config.MODE!r}.")

# test documents per author; Reuters is ~54s per document (512 tokens), darkreddit ~8s
DOCS_PER_AUTHOR = {"darkreddit": 5, "reuters": 1}
MAX_K = 99
# every n needs at least n content tokens to have a window
MIN_CONTENT_TOKENS = max(NGRAM_SIZES)
# wall time from script start after which no new document is started (devel jobs get 10 min)
TIME_BUDGET_SECONDS = 8 * 60
# seconds per (sequence token x embedded variant), doc_mean + plain on the A100 (outputs/masking.log);
# replaced by this run's own measurement after the first document
INITIAL_SECONDS_PER_TOKEN_VARIANT = 0.06e-3

SETTINGS_INFO = {
    "docs_per_author": DOCS_PER_AUTHOR[config.DATASET],
    "max_k": MAX_K,
    "ngram_sizes": NGRAM_SIZES,
    "combine": "token_rank_average",
    "selection": "non_overlapping_windows_then_fill",
    "min_content_tokens": MIN_CONTENT_TOKENS,
}


def plan_documents(authors):
    # first docs_per_author test documents per author with at least MIN_CONTENT_TOKENS content
    # tokens, interleaved (every author's first, then every author's second, ...) so a series
    # that is cut short is still balanced over authors
    per_author = {}
    skipped = []
    for author in authors:
        picked = []
        for doc_index, text in enumerate(load_test_texts(config.DATASET, author)):
            if len(picked) == SETTINGS_INFO["docs_per_author"]:
                break
            if content_token_count(text) < MIN_CONTENT_TOKENS:
                skipped.append([author, doc_index])
                continue
            picked.append(doc_index)
        per_author[author] = picked

    plan = [
        [author, per_author[author][i]]
        for i in range(SETTINGS_INFO["docs_per_author"])
        for author in authors
        if i < len(per_author[author])
    ]
    return plan, skipped


def load_saved_scores(author, doc_index, n, tokens):
    # occlusion scores from an earlier run.py run with the same dataset, perturbation, model and
    # mode (its folder), if it exists and tokenized this document the same way
    path = os.path.join(config.MASKING_RESULTS_DIR, f"{author}_{doc_index}_n{n}.json")
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        saved = json.load(f)
    if saved["tokens"] != tokens:
        return None, None
    return saved["scores"], path


def similarities_and_ranks(embeddings, centroid_matrix, true_index):
    # cosine similarity to the true author's centroid, and the true author's rank among all
    # centroids (1 = retrieved first), ranked the same way retrieval.mean_avg_precision does
    sims = torch.nn.functional.cosine_similarity(
        embeddings.unsqueeze(1), centroid_matrix.unsqueeze(0), dim=2
    )
    order = sims.argsort(dim=1, descending=True)
    ranks = (order == true_index).nonzero()[:, 1] + 1
    return sims[:, true_index].tolist(), ranks.tolist()


def estimated_work(num_tokens, scores_saved):
    # embedded variants x sequence length, the unit INITIAL_SECONDS_PER_TOKEN_VARIANT is per
    content = num_tokens - 2
    scoring_variants = 0 if scores_saved else len(NGRAM_SIZES) * content
    keep_variants = len(SETTINGS) * min(MAX_K, content - 1) + 1
    return num_tokens * (scoring_variants + keep_variants)


def run_document(author, doc_index, centroids, centroid_matrix, authors):
    centroid = centroids[author]
    true_index = authors.index(author)
    text = load_test_texts(config.DATASET, author)[doc_index]
    input_ids, attention_mask = tokenize(text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    num_tokens = input_ids.shape[1]
    num_content = num_tokens - 2
    all_positions = list(range(num_tokens))

    base_embedding = embed_variants(input_ids, attention_mask, [all_positions])
    base_sims, base_ranks = similarities_and_ranks(base_embedding, centroid_matrix, true_index)
    baseline = {"similarity": base_sims[0], "rank": base_ranks[0]}

    scores_by_n = {}
    scores_source = {}
    for n in NGRAM_SIZES:
        scores, path = load_saved_scores(author, doc_index, n, tokens)
        if scores is None:
            _, scores = occlusion_scores(input_ids, attention_mask, centroid, n)
            path = "computed"
        scores_by_n[n] = scores
        scores_source[str(n)] = path

    # k < num_content changes the document; at k >= num_content it stays whole (= baseline)
    changed_ks = [k for k in range(1, MAX_K + 1) if k < num_content]
    combined_order = rank_average_order(scores_by_n, num_tokens)

    keep = {}
    for setting in SETTINGS:
        kept_positions = []
        filled = []
        if setting == COMBINED:
            for k in changed_ks:
                kept_positions.append(sorted(combined_order[:k]))
                filled.append(None)
        else:
            n = int(setting[1:])
            ranking = window_ranking(scores_by_n[n], num_tokens, n)
            for k in changed_ks:
                positions, was_filled = keep_top_windows(ranking, n, k)
                kept_positions.append(positions)
                filled.append(was_filled)

        for positions, k in zip(kept_positions, changed_ks):
            assert len(positions) == k and all(0 < p < num_tokens - 1 for p in positions)

        entries = []
        if changed_ks:
            keep_lists = [[0] + positions + [num_tokens - 1] for positions in kept_positions]
            embeddings = embed_variants(input_ids, attention_mask, keep_lists)
            sims, ranks = similarities_and_ranks(embeddings, centroid_matrix, true_index)
            for k, sim, rank, was_filled in zip(changed_ks, sims, ranks, filled):
                entry = {"k": k, "similarity": sim, "rank": rank, "unchanged": False}
                if was_filled is not None:
                    entry["filled"] = was_filled
                entries.append(entry)
        for k in range(len(changed_ks) + 1, MAX_K + 1):
            entries.append({"k": k, "similarity": baseline["similarity"], "rank": baseline["rank"], "unchanged": True})
        keep[setting] = entries

    return {
        "author": author,
        "doc_index": doc_index,
        "true_author_index": true_index,
        # content tokens the model sees (after truncation to 512 incl. <s></s>), and before truncation
        "num_content_tokens": num_content,
        "full_content_tokens": content_token_count(text, max_tokens=10**9),
        "baseline": baseline,
        "keep": keep,
        "scores_source": scores_source,
        "tokens": tokens,
        "scores": {str(n): scores_by_n[n] for n in NGRAM_SIZES},
    }


def print_summary(documents):
    summary = aggregate(documents, MAX_K)
    print(f"\n{summary['num_documents']} documents in this series so far")
    base = summary["baseline"]
    print(f"baseline (full document): similarity {base['similarity']:.4f}, top-1 {base['top1']:.2%}, MAP {base['map']:.2%}")
    print("setting | k  | similarity | top-1   | MAP     | unchanged docs")
    for setting in SETTINGS:
        for k in (1, 2, 5, 10, 20, 50, MAX_K):
            row = summary["curves"][setting][k - 1]
            print(f"{setting:<7} | {k:<2} | {row['similarity']:.4f}     | {row['top1']:6.2%} | {row['map']:6.2%} | {row['num_unchanged']}")


def main():
    started_at = datetime.now()
    print(f"dataset: {config.DATASET}, mode: {config.MODE}, model: {config.MODEL}, perturbation: {config.PERTURBATION}")

    authors = load_authors(config.DATASET)
    if config.MODE == "chunk32":
        centroids = get_centroids(config.DATASET, authors, embed_fn=embed_chunked, cache_name="centroids_chunk32.pt")
    else:
        centroids = get_centroids(config.DATASET, authors)
    centroid_matrix = torch.stack([centroids[a] for a in authors])

    plan, skipped = plan_documents(authors)
    out_dir = results_dir(config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME)
    os.makedirs(out_dir, exist_ok=True)

    # continue the latest series if it has the same settings and plan and isn't complete yet
    series, series_runs = latest_series(load_runs(out_dir))
    done_docs = series_documents(series_runs)
    last = series_runs[-1] if series_runs else None
    if last is None or last["settings"] != SETTINGS_INFO or last["plan"] != plan or len(done_docs) >= len(plan):
        series = f"{started_at:%Y%m%d-%H%M%S}"
        series_runs = []
        done_docs = []
        print(f"starting new series {series}")
    else:
        print(f"continuing series {series}: {len(done_docs)}/{len(plan)} documents done")

    done = {(doc["author"], doc["doc_index"]) for doc in done_docs}
    todo = [(author, doc_index) for author, doc_index in plan if (author, doc_index) not in done]

    result_path = os.path.join(out_dir, f"keep_{started_at:%Y%m%d-%H%M%S}.json")
    result = {
        "experiment": "keep_top_k",
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

    seconds_per_unit = INITIAL_SECONDS_PER_TOKEN_VARIANT
    total_seconds = 0.0
    total_work = 0
    for author, doc_index in todo:
        text = load_test_texts(config.DATASET, author)[doc_index]
        input_ids, _ = tokenize(text)
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        scores_saved = all(load_saved_scores(author, doc_index, n, tokens)[0] is not None for n in NGRAM_SIZES)
        work = estimated_work(input_ids.shape[1], scores_saved)
        elapsed = time.time() - script_start
        if elapsed + work * seconds_per_unit > TIME_BUDGET_SECONDS:
            print(f"\ntime budget: {elapsed:.0f}s used, next document estimated {work * seconds_per_unit:.0f}s - stopping, run again to continue")
            break

        doc_start = time.time()
        doc = run_document(author, doc_index, centroids, centroid_matrix, authors)
        doc["seconds"] = time.time() - doc_start
        total_seconds += doc["seconds"]
        total_work += work
        seconds_per_unit = total_seconds / total_work

        print(f"{author}, doc {doc_index}, {doc['num_content_tokens']} tokens: {doc['seconds']:.1f}s, "
              f"baseline {doc['baseline']['similarity']:.4f} (rank {doc['baseline']['rank']}), "
              f"k=10: " + ", ".join(f"{s} {doc['keep'][s][9]['similarity']:.4f}" for s in SETTINGS))

        # saved after every document so a job killed at its time limit loses at most one
        result["documents"].append(doc)
        result["complete"] = len(done_docs) + len(result["documents"]) >= len(plan)
        with open(result_path, "w") as f:
            json.dump(result, f)

    all_docs = done_docs + result["documents"]
    print(f"\nseries {series}: {len(all_docs)}/{len(plan)} documents done"
          + ("" if len(all_docs) >= len(plan) else " - run again to continue"))
    if result["documents"]:
        print(f"saved results to {result_path}")
    if all_docs:
        print_summary(all_docs)


if __name__ == "__main__":
    main()
