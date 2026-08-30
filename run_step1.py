import os
import torch
from luar_model import embed, embed_episode

def load_author_texts(folder):
    texts = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts

authors = os.listdir("data/C50train")

centroids = {}
supports = {}
for i in range(len(os.listdir("data/C50train"))):
    print("calculating centroid for", authors[i])
    supports[authors[i]] = load_author_texts(f"data/C50train/{authors[i]}")
    centroids[authors[i]] = embed_episode(supports[authors[i]])

correct = 0
total = 0
per_author_correct = {}
per_author_total = {}

for i in range(len(authors)):
    query_texts = load_author_texts(f"data/C50test/{authors[i]}")
    author_correct = 0
    for query_text in query_texts:
        query_embedding = embed(query_text)
        sims = {}
        for j in range(len(authors)):
            sims[j] = torch.nn.functional.cosine_similarity(query_embedding, centroids[authors[j]], dim=0)
        predicted = max(sims, key=sims.get)
        if predicted == i:
            author_correct += 1
            correct += 1
        total += 1

    per_author_correct[authors[i]] = author_correct
    per_author_total[authors[i]] = len(query_texts)
    print(f"{authors[i]}: {author_correct}/{len(query_texts)}")

print(f"\ntotal accuracy: {correct}/{total}")