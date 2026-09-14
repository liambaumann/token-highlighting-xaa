import os
import torch
from luar_model import embed_from_ids, tokenizer
from reuters_data import load_author_texts, get_centroids
from highlight import save_highlighted_html

authors = sorted(
    name for name in os.listdir("data/C50train")
    if os.path.isdir(os.path.join("data/C50train", name))
)
centroids = get_centroids(authors)

# Test single author only
author = authors[0]
# load first text by that author
query_text = load_author_texts(f"data/C50test/{author}")[0]
# get centroid of first author
author_centroid = centroids[author]

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
    

print(f"base similarity: {base_similarity:.4f}")
print(f"author: {author}")
print(f"token count: {input_ids.shape[1]}")

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

for token, score in zip(tokens, scores):
    print(f"{score:+.4f}  {token}")

top_index = max(range(len(scores)), key=lambda i: scores[i])
print(f"\ntop token: '{tokens[top_index]}'")

top3_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:3]
keep_mask = [i for i in range(input_ids.shape[1]) if i not in top3_indices]

masked_ids = input_ids[:, keep_mask]
masked_attention = attention_mask[:, keep_mask]

masked_similarity = torch.nn.functional.cosine_similarity(
    embed_from_ids(masked_ids, masked_attention), author_centroid, dim=0
).item()

print(f"\nbase similarity: {base_similarity:.4f}")
print(f"similarity after removing top 3 tokens: {masked_similarity:.4f}")
print(f"drop: {base_similarity - masked_similarity:+.4f}")

filepath = save_highlighted_html(tokens, scores, author, tokenizer)
print(f"\nsaved highlighted output to {filepath}")