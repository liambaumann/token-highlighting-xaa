import os
import torch
from luar_model import embed

def load_author_texts(folder):
    texts = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts

def centroid(texts):
    embeddings = torch.stack([embed(t) for t in texts])
    return embeddings.mean(dim=0)

authors = os.listdir("data/C50train")

centroids = {}
supports = {}
for i in range(len(os.listdir("data/C50train"))):
    print("calculating centroid for", authors[i])
    supports[authors[i]] = load_author_texts(f"data/C50train/{authors[i]}")
    centroids[authors[i]] = centroid(supports[authors[i]])

correct = 0
for i in range(len(authors)):
    print("testing", authors[i])
    query_embedding = embed(load_author_texts(f"data/C50test/{authors[i]}")[0])
    sims = {}
    for j in range(len(authors)):
        sims[j] = torch.nn.functional.cosine_similarity(query_embedding, centroids[authors[j]], dim=0)
    predicted = max(sims, key=sims.get)
    if predicted == i:
        correct += 1
    print(authors[i], "-> predicted:", authors[predicted], "correct:" if predicted == i else "wrong:")

print(f"\naccuracy: {correct}/{len(authors)}")