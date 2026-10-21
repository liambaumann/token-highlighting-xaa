# keep-top-k selection: which content positions of a document survive when only its k
# highest-scoring tokens / n-grams are kept. Works on the score lists from
# scoring.occlusion_scores, no model needed. Also the aggregation of saved keep-top-k results
# (metrics, aggregate), so keep_top_k.plot can use it without loading LUAR.

RESULTS_ROOT = "results/keep_top_k"

NGRAM_SIZES = [1, 2, 3]
COMBINED = "n1-3"
SETTINGS = [f"n{n}" for n in NGRAM_SIZES] + [COMBINED]


def valid_starts(num_tokens, n):
    # the windows scoring.occlusion_scores scores: never covering <s> (0) or </s> (num_tokens - 1)
    return list(range(1, num_tokens - n))


def window_ranking(scores, num_tokens, n):
    # highest score first; ties keep position order (sorted is stable, also with reverse=True)
    return sorted(valid_starts(num_tokens, n), key=lambda s: scores[s], reverse=True)


def keep_top_windows(ranking, n, k):
    # non-overlapping windows in rank order (as masking.windows.select_windows) until k tokens are kept,
    # the window that crosses k cropped to its leftmost tokens. Greedy non-overlapping windows
    # can leave gaps shorter than n, so if they run out before k, continue down the same ranking
    # with the still-free tokens of partly overlapping windows.
    # returns (sorted kept positions, whether that fill-up step was needed)
    kept = set()
    for start in ranking:
        window = range(start, start + n)
        if any(p in kept for p in window):
            continue
        kept.update(list(window)[:k - len(kept)])
        if len(kept) == k:
            return sorted(kept), False

    for start in ranking:
        free = [p for p in range(start, start + n) if p not in kept]
        kept.update(free[:k - len(kept)])
        if len(kept) == k:
            break
    return sorted(kept), True


def token_scores(scores, num_tokens, n):
    # per content position: mean score of the n-gram windows covering it
    sums = {p: 0.0 for p in range(1, num_tokens - 1)}
    counts = {p: 0 for p in range(1, num_tokens - 1)}
    for start in valid_starts(num_tokens, n):
        for p in range(start, start + n):
            sums[p] += scores[start]
            counts[p] += 1
    return {p: sums[p] / counts[p] for p in sums}


def rank_average_order(scores_by_n, num_tokens):
    # combined 1-3 ranking: for each n, rank the content tokens by token_scores (1 = best),
    # then order tokens by their mean rank over all n (ties keep position order)
    positions = list(range(1, num_tokens - 1))
    rank_sums = {p: 0 for p in positions}
    for n, scores in scores_by_n.items():
        per_token = token_scores(scores, num_tokens, n)
        order = sorted(positions, key=lambda p: per_token[p], reverse=True)
        for rank, p in enumerate(order, start=1):
            rank_sums[p] += rank
    # same order as the mean rank, every token is summed over the same number of n
    return sorted(positions, key=lambda p: rank_sums[p])


def metrics(entries):
    # entries: {"similarity", "rank"} per document - top-1 and MAP as in metrics.py
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
