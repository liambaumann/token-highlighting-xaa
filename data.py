import os
import torch
from model import embed_episode
import config

DATASETS = {"reuters": {"train": "data/C50train", "test": "data/C50test"}}


def load_authors(dataset):
    train_dir = DATASETS[dataset]["train"]
    return sorted(
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    )


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


def load_train_texts(dataset, author):
    return load_author_texts(os.path.join(DATASETS[dataset]["train"], author))


def load_test_texts(dataset, author):
    return load_author_texts(os.path.join(DATASETS[dataset]["test"], author))


def get_centroids(dataset, authors):
    cache_path = os.path.join(config.CACHE_DIR, "centroids.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    centroids = {}
    for author in authors:
        print("calculating centroid for", author)
        support_texts = load_train_texts(dataset, author)
        centroids[author] = embed_episode(support_texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(centroids, cache_path)
    return centroids
