import os
import json
import random
import statistics
import torch
from model import embed, embed_chunked, embed_single, content_token_count, tokenizer
from data import load_authors, load_train_texts, load_test_texts, get_centroids
import config

# from earlier full 50-author runs (see retrieval_results.md), for the final comparison table
BASELINE_RESULTS = {
    "plain": {"top1": 0.5124, "top5": 0.8920, "top10": 0.9700, "map": 0.6733},
    "chunk32": {"top1": 0.5620, "top5": 0.8900, "top10": 0.9704, "map": 0.7039},
}


def get_similarity_matrix(authors, centroids, embed_fn=embed):
    sims = []
    true_labels = []
    for i, author in enumerate(authors):
        print("testing", author)
        query_texts = load_test_texts(config.DATASET, author)
        for query_text in query_texts:
            query_embedding = embed_fn(query_text)
            row = [
                torch.nn.functional.cosine_similarity(query_embedding, centroids[a], dim=0).item()
                for a in authors
            ]
            sims.append(row)
            true_labels.append(i)

    sims = torch.tensor(sims)
    true_labels = torch.tensor(true_labels)
    return sims, true_labels


def top1_accuracy(sims, true_labels):
    predictions = sims.argmax(dim=1)
    correct = (predictions == true_labels).sum().item()
    total = len(true_labels)
    return correct, total


def soft_top_x_accuracy(sims, true_labels, x):
    top_x_predictions = sims.topk(x, dim=1).indices
    correct = sum(true_labels[i] in top_x_predictions[i] for i in range(len(true_labels)))
    total = len(true_labels)
    return correct, total


def mean_avg_precision(sims, true_labels):
    reciprocal_ranks = []
    for i in range(len(true_labels)):
        row = sims[i]
        true_col = true_labels[i].item()
        sorted_indices = row.argsort(descending=True)
        rank = (sorted_indices == true_col).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def print_retrieval_metrics(sims, true_labels):
    correct, total = top1_accuracy(sims, true_labels)
    correct_top5, total_top5 = soft_top_x_accuracy(sims, true_labels, 5)
    correct_top10, total_top10 = soft_top_x_accuracy(sims, true_labels, 10)
    mAP = mean_avg_precision(sims, true_labels)
    print(f"\ntop-1 accuracy: {correct}/{total} (~{correct/total:.2%})")
    print(f"soft top-5 accuracy: {correct_top5}/{total_top5} (~{correct_top5/total_top5:.2%})")
    print(f"soft top-10 accuracy: {correct_top10}/{total_top10} (~{correct_top10/total_top10:.2%})")
    print(f"mean average precision: {mAP:.2%}")


def pick_excerpt_start(rng, text):
    num_content = content_token_count(text)
    max_start = max(0, num_content - 30)
    return rng.randint(0, max_start)


def decode_single32_excerpt(rng, text):
    start = pick_excerpt_start(rng, text)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:510]
    piece_ids = [tokenizer.bos_token_id] + ids[start:start+30] + [tokenizer.eos_token_id]
    print("example single32 excerpt:", tokenizer.decode(piece_ids))
    print("token count:", len(piece_ids))


def run_single32_seed(seed, authors):
    rng = random.Random(seed)

    # precompute all excerpt starts up front, in the same order get_centroids/get_similarity_matrix
    # consume them, so the sequence of random draws (and the cached-centroid case) stays reproducible
    centroid_starts = {}
    for author in authors:
        support_texts = load_train_texts(config.DATASET, author)
        centroid_starts[author] = [pick_excerpt_start(rng, text) for text in support_texts]

    query_starts = {}
    for author in authors:
        query_texts = load_test_texts(config.DATASET, author)
        query_starts[author] = [pick_excerpt_start(rng, text) for text in query_texts]

    centroid_author_iter = iter(authors)
    def centroid_embed_fn(texts):
        author = next(centroid_author_iter)
        return embed_single(texts, centroid_starts[author])

    query_start_iter = iter(start for author in authors for start in query_starts[author])
    def query_embed_fn(text):
        start = next(query_start_iter)
        return embed_single([text], [start])

    centroids = get_centroids(config.DATASET, authors, embed_fn=centroid_embed_fn, cache_name=f"centroids_single32_seed{seed}.pt")
    sims, true_labels = get_similarity_matrix(authors, centroids, embed_fn=query_embed_fn)

    correct, total = top1_accuracy(sims, true_labels)
    correct_top5, total_top5 = soft_top_x_accuracy(sims, true_labels, 5)
    correct_top10, total_top10 = soft_top_x_accuracy(sims, true_labels, 10)
    mAP = mean_avg_precision(sims, true_labels)

    metrics = {
        "top1": {"correct": correct, "total": total},
        "soft_top5": {"correct": correct_top5, "total": total_top5},
        "soft_top10": {"correct": correct_top10, "total": total_top10},
        "map": mAP,
    }

    os.makedirs(config.RETRIEVAL_RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(config.RETRIEVAL_RESULTS_DIR, f"seed{seed}.json")
    with open(result_path, "w") as f:
        json.dump({
            "dataset": config.DATASET,
            "model": config.MODEL,
            "mode": "single32",
            "seed": seed,
            "metrics": metrics,
            "centroid_excerpt_starts": centroid_starts,
            "query_excerpt_starts": query_starts,
        }, f, indent=2)
    print(f"saved results to {result_path}")

    return metrics


if __name__ == "__main__":
    authors = load_authors(config.DATASET)

    if config.MODE == "single32":
        decode_single32_excerpt(random.Random(config.SINGLE32_SEEDS[0]), load_train_texts(config.DATASET, authors[0])[0])

        seed_metrics = []
        for seed in config.SINGLE32_SEEDS:
            print(f"\n--- single32, seed {seed} ---")
            metrics = run_single32_seed(seed, authors)
            seed_metrics.append(metrics)
            print(f"top-1 accuracy: {metrics['top1']['correct']}/{metrics['top1']['total']} "
                  f"(~{metrics['top1']['correct']/metrics['top1']['total']:.2%})")
            print(f"soft top-5 accuracy: {metrics['soft_top5']['correct']}/{metrics['soft_top5']['total']} "
                  f"(~{metrics['soft_top5']['correct']/metrics['soft_top5']['total']:.2%})")
            print(f"soft top-10 accuracy: {metrics['soft_top10']['correct']}/{metrics['soft_top10']['total']} "
                  f"(~{metrics['soft_top10']['correct']/metrics['soft_top10']['total']:.2%})")
            print(f"mean average precision: {metrics['map']:.2%}")

        def rate(m, key):
            return m[key]["correct"] / m[key]["total"]

        single32_summary = {
            "top1": (statistics.mean(rate(m, "top1") for m in seed_metrics), statistics.stdev(rate(m, "top1") for m in seed_metrics)),
            "top5": (statistics.mean(rate(m, "soft_top5") for m in seed_metrics), statistics.stdev(rate(m, "soft_top5") for m in seed_metrics)),
            "top10": (statistics.mean(rate(m, "soft_top10") for m in seed_metrics), statistics.stdev(rate(m, "soft_top10") for m in seed_metrics)),
            "map": (statistics.mean(m["map"] for m in seed_metrics), statistics.stdev(m["map"] for m in seed_metrics)),
        }

        print("\nmode                              | top-1          | soft top-5     | soft top-10    | MAP")
        for name, r in BASELINE_RESULTS.items():
            print(f"{name:<34} | {r['top1']:.2%}         | {r['top5']:.2%}         | {r['top10']:.2%}         | {r['map']:.2%}")
        print(f"{'single32 (mean ± std, seeds ' + str(config.SINGLE32_SEEDS) + ')':<34} | "
              f"{single32_summary['top1'][0]:.2%} ± {single32_summary['top1'][1]:.2%} | "
              f"{single32_summary['top5'][0]:.2%} ± {single32_summary['top5'][1]:.2%} | "
              f"{single32_summary['top10'][0]:.2%} ± {single32_summary['top10'][1]:.2%} | "
              f"{single32_summary['map'][0]:.2%} ± {single32_summary['map'][1]:.2%}")

    elif config.MODE == "chunk32":
        centroids = get_centroids(config.DATASET, authors, embed_fn=embed_chunked, cache_name="centroids_chunk32.pt")
        sims, true_labels = get_similarity_matrix(authors, centroids, embed_fn=lambda t: embed_chunked([t]))
        print_retrieval_metrics(sims, true_labels)

    elif config.MODE == "plain":
        centroids = get_centroids(config.DATASET, authors)
        sims, true_labels = get_similarity_matrix(authors, centroids)
        print_retrieval_metrics(sims, true_labels)

    else:
        raise ValueError(f"unknown config.MODE: {config.MODE!r}")
