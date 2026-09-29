import os
import torch
from luar_model import embed_episode

"""
Reutrn list of full texts of each author folder
"""
def load_author_texts(folder):
    texts = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        with open(path, encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def get_centroids(authors, cache_path="cache/centroids.pt"):
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    centroids = {}
    for author in authors:
        print("calculating centroid for", author)
        support_texts = load_author_texts(f"data/C50train/{author}")
        centroids[author] = embed_episode(support_texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(centroids, cache_path)
    return centroids