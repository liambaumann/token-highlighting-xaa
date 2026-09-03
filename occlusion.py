import os
import torch
from luar_model import embed
from reuters_data import load_author_texts, get_centroids
from luar_model import tokenizer

authors = sorted(os.listdir("data/C50train"))
centroids = get_centroids(authors)

author = authors[0]
query_text = load_author_texts(f"data/C50test/{author}")[0]

author_centroid = centroids[author]
query_words = query_text.split()

cutoff = len(query_words)
for i in range(len(query_words)):
    token_count = len(tokenizer(" ".join(query_words[:i + 1]))["input_ids"])
    if token_count > 512:
        cutoff = i
        break
query_words = query_words[:cutoff]

reconstructed_query = " ".join(query_words)
base_similarity = torch.nn.functional.cosine_similarity(
    embed(reconstructed_query), author_centroid, dim=0
).item()

scores = []
for i in range(len(query_words)):
    perturbed_text = " ".join(query_words[:i] + query_words[i + 1:])
    perturbed_similarity = torch.nn.functional.cosine_similarity(
        embed(perturbed_text), author_centroid, dim=0
    ).item()
    scores.append(base_similarity - perturbed_similarity)

print(f"author: {author}")
print(f"base similarity: {base_similarity:.4f}\n")

for word, score in zip(query_words, scores):
    print(f"{score:+.4f}  {word}")

top_index = max(range(len(scores)), key=lambda i: scores[i])
print(f"\ntop token: '{query_words[top_index]}'")

top_k = 10
ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
top_indices = set(ranked[:top_k])

highlighted = " ".join(
    f"[{w}]" if i in top_indices else w for i, w in enumerate(query_words)
)
print(f"\nhighlighted query:\n{highlighted}")