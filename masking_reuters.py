import os
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
        embed_from_ids(input_ids, attention_mask), author_centroid, dim=0
    ).item()

    num_tokens = input_ids.shape[1]
    scores = []

    for i in range(num_tokens):
        perturbed_ids = torch.cat([input_ids[:, :i], input_ids[:, i+1:]], dim=1)
        perturbed_mask = torch.cat([attention_mask[:, :i], attention_mask[:, i+1:]], dim=1)
        perturbed_similarity = torch.nn.functional.cosine_similarity(
            embed_from_ids(perturbed_ids, perturbed_mask), author_centroid, dim=0
        ).item()
        scores.append(base_similarity - perturbed_similarity)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    budget_results = []
    for pct in budget_percentages:
        k = max(1, round(pct * num_tokens))

        top_indices = sorted_indices[:k]
        keep_mask = [i for i in range(input_ids.shape[1]) if i not in top_indices]
        masked_ids = input_ids[:, keep_mask]
        masked_attention = attention_mask[:, keep_mask]
        top_similarity = torch.nn.functional.cosine_similarity(
            embed_from_ids(masked_ids, masked_attention), author_centroid, dim=0
        ).item()

        random_indices = random.sample(range(input_ids.shape[1]), k)
        keep_mask_random = [i for i in range(input_ids.shape[1]) if i not in random_indices]
        random_ids = input_ids[:, keep_mask_random]
        random_attention = attention_mask[:, keep_mask_random]
        random_similarity = torch.nn.functional.cosine_similarity(
            embed_from_ids(random_ids, random_attention), author_centroid, dim=0
        ).item()

        budget_results.append({
            "k": k,
            "pct": pct,
            "top_similarity": top_similarity,
            "random_similarity": random_similarity,
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
    (authors[1], 0),
    (authors[2], 0),
]

all_charts_html = ""

for author, doc_index in test_cases:
    author_centroid = centroids[author]
    query_text = load_author_texts(f"data/C50test/{author}")[doc_index]

    result = run_occlusion(query_text, author, author_centroid)

    filename = f"{author}_{doc_index}"
    highlight_path = save_highlighted_html(result["tokens"], result["scores"], filename, tokenizer)

    print(f"\n{author}, doc {doc_index}")
    print(f"base similarity: {result['base_similarity']:.4f}")
    for budget in result["budget_results"]:
        print(f"top-{budget['k']} removed ({budget['pct']:.0%}): {budget['top_similarity']:.4f}")
        print(f"random-{budget['k']} removed ({budget['pct']:.0%}): {budget['random_similarity']:.4f}")

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

with open("outputs/overview.html", "w") as f:
    f.write(overview_html)

print("\nsaved overview to outputs/overview.html")