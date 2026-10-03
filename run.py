import random
random.seed(0)

import os
import glob
import json
import time
import config
from model import tokenizer, embed_chunked
from data import load_authors, load_test_texts, get_centroids
from scoring import tokenize, occlusion_scores
from masking import evaluate_budgets

if config.MODE not in ("plain", "chunk32"):
    raise ValueError(f"run.py supports MODE \"plain\" and \"chunk32\", not {config.MODE!r}. single32 is retrieval only (retrieval.py).")

budget_percentages = [0.01, 0.03, 0.05, 0.10]
ngram_sizes = [1, 2, 3]

os.makedirs(config.MASKING_RESULTS_DIR, exist_ok=True)
for old_file in glob.glob(os.path.join(config.MASKING_RESULTS_DIR, "*.json")):
    os.remove(old_file)

authors = load_authors(config.DATASET)
if config.MODE == "chunk32":
    centroids = get_centroids(config.DATASET, authors, embed_fn=embed_chunked, cache_name="centroids_chunk32.pt")
else:
    centroids = get_centroids(config.DATASET, authors)

test_cases = [
    #(authors[0], 2),
    #(authors[1], 0),
    #(authors[2], 0),
    #(authors[3], 0),
    #(authors[4], 0),
] + [(author, 0) for author in authors[5:50]]

for author, doc_index in test_cases:
    author_centroid = centroids[author]
    query_text = load_test_texts(config.DATASET, author)[doc_index]
    input_ids, attention_mask = tokenize(query_text)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    for n in ngram_sizes:
        start_time = time.time()

        base_similarity, scores = occlusion_scores(input_ids, attention_mask, author_centroid, n)
        budget_results = evaluate_budgets(input_ids, attention_mask, author_centroid, scores, budget_percentages, n)

        result = {
            "author": author,
            "tokens": tokens,
            "scores": scores,
            "base_similarity": base_similarity,
            "budget_results": budget_results,
        }

        print(f"\n{author}, doc {doc_index}, n={n}: {time.time() - start_time:.2f}s")

        print(f"base similarity: {result['base_similarity']:.4f}")
        for budget in result["budget_results"]:
            print(f"top-{budget['k']} removed ({budget['pct']:.0%}): {budget['top_similarity']:.4f}")
            print(f"random-{budget['k']} removed ({budget['pct']:.0%}): avg {budget['random_similarity_avg']:.4f} {[round(s, 4) for s in budget['random_similarities']]}")
            print(f"bottom-{budget['k']} removed ({budget['pct']:.0%}): {budget['bottom_similarity']:.4f}")

        result_path = os.path.join(config.MASKING_RESULTS_DIR, f"{author}_{doc_index}_n{n}.json")
        with open(result_path, "w") as f:
            json.dump({
                "author": author,
                "doc_index": doc_index,
                "n": n,
                "base_similarity": result["base_similarity"],
                "budget_results": result["budget_results"],
                "tokens": result["tokens"],
                "scores": result["scores"],
            }, f, indent=2)
