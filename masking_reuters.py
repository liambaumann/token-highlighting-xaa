import os
import time
import torch
import random
from luar_model import embed_from_ids, tokenizer
from reuters_data import load_author_texts, get_centroids
from highlight import save_highlighted_html
from chart import build_chart_html


def run_occlusion(query_text, author, author_centroid, budget_percentages=[0.01, 0.03, 0.05, 0.10]):
    query_text_no_newlines = " ".join(query_text.split())
    tokenized = tokenizer(query_text_no_newlines, max_length=512, truncation=True, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    base_similarity = torch.nn.functional.cosine_similarity(
        embed_from_ids(input_ids, attention_mask), author_centroid.unsqueeze(0), dim=1
    ).item()

    num_tokens = input_ids.shape[1]

    loo_ids = torch.cat([
        torch.cat([input_ids[:, :i], input_ids[:, i+1:]], dim=1)
        for i in range(num_tokens)
    ], dim=0)
    loo_mask = torch.cat([
        torch.cat([attention_mask[:, :i], attention_mask[:, i+1:]], dim=1)
        for i in range(num_tokens)
    ], dim=0)
    loo_embeddings = embed_from_ids(loo_ids, loo_mask)
    perturbed_similarities = torch.nn.functional.cosine_similarity(
        loo_embeddings, author_centroid.unsqueeze(0), dim=1
    )
    scores = (base_similarity - perturbed_similarities).tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    budget_results = []
    for pct in budget_percentages:
        k = max(1, round(pct * num_tokens))

        top_indices = sorted_indices[:k]
        keep_masks = [[i for i in range(input_ids.shape[1]) if i not in top_indices]]

        for _ in range(5):
            random_indices = random.sample(range(input_ids.shape[1]), k)
            keep_masks.append([i for i in range(input_ids.shape[1]) if i not in random_indices])

        batch_ids = torch.cat([input_ids[:, keep_mask] for keep_mask in keep_masks], dim=0)
        batch_attention = torch.cat([attention_mask[:, keep_mask] for keep_mask in keep_masks], dim=0)
        batch_embeddings = embed_from_ids(batch_ids, batch_attention)
        similarities = torch.nn.functional.cosine_similarity(
            batch_embeddings, author_centroid.unsqueeze(0), dim=1
        ).tolist()

        top_similarity = similarities[0]
        random_similarities = similarities[1:]

        budget_results.append({
            "k": k,
            "pct": pct,
            "top_similarity": top_similarity,
            "random_similarities": random_similarities,
            "random_similarity_avg": sum(random_similarities) / len(random_similarities),
        })

    return {
        "author": author,
        "tokens": tokens,
        "scores": scores,
        "base_similarity": base_similarity,
        "budget_results": budget_results,
    }


authors = sorted(
    name for name in os.listdir("data/C50train")
    if os.path.isdir(os.path.join("data/C50train", name))
)
centroids = get_centroids(authors)

test_cases = [
    (authors[0], 2),
    #(authors[1], 0),
    #(authors[2], 0),
]

all_charts_html = ""

for author, doc_index in test_cases:
    author_centroid = centroids[author]
    query_text = load_author_texts(f"data/C50test/{author}")[doc_index]

    start_time = time.time()
    result = run_occlusion(query_text, author, author_centroid)
    print(f"\n{author}, doc {doc_index}: {time.time() - start_time:.2f}s")

    filename = f"{author}_{doc_index}"
    highlight_path = save_highlighted_html(result["tokens"], result["scores"], filename, tokenizer)

    print(f"base similarity: {result['base_similarity']:.4f}")
    for budget in result["budget_results"]:
        print(f"top-{budget['k']} removed ({budget['pct']:.0%}): {budget['top_similarity']:.4f}")
        print(f"random-{budget['k']} removed ({budget['pct']:.0%}): avg {budget['random_similarity_avg']:.4f} {[round(s, 4) for s in budget['random_similarities']]}")

    chart_title = f"{author}_{doc_index}"
    all_charts_html += build_chart_html(
        result["base_similarity"], result["budget_results"],
        title=chart_title, link_href=os.path.basename(highlight_path),
    )

overview_html = f"""<html>
<head><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script></head>
<body>
{all_charts_html}
</body>
</html>"""

with open("outputs/masking_reuters_overview.html", "w") as f:
    f.write(overview_html)

print("\nsaved overview to outputs/masking_reuters_overview.html")