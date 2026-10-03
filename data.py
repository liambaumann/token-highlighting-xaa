import os
import json
import random
import torch
from model import embed_episode
import config

DATASETS = {
    "reuters": {"type": "folders", "train": "data/reuter_50_50/C50train", "test": "data/reuter_50_50/C50test"},
    "darkreddit": {
        "type": "jsonl",
        "train": "data/darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_train_anon.jsonl",
        "test": "data/darkreddit_authorship_attribution_anon/darkreddit_authorship_attribution_test_anon.jsonl",
    },
}

DARKREDDIT_TRAIN_SAMPLE_SIZE = 50
DARKREDDIT_SAMPLE_SEED = 0


def load_authors(dataset):
    if DATASETS[dataset]["type"] == "folders":
        train_dir = DATASETS[dataset]["train"]
        return sorted(
            name for name in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, name))
        )
    else:
        return sorted(load_jsonl_by_author(DATASETS[dataset]["train"]).keys())


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


def load_jsonl_by_author(path):
    by_author = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            by_author.setdefault(row["author"], []).append(row["comment"])
    return by_author


def load_train_texts(dataset, author):
    if DATASETS[dataset]["type"] == "folders":
        return load_author_texts(os.path.join(DATASETS[dataset]["train"], author))

    comments = load_jsonl_by_author(DATASETS[dataset]["train"])[author]
    rng = random.Random(DARKREDDIT_SAMPLE_SEED)
    return rng.sample(comments, DARKREDDIT_TRAIN_SAMPLE_SIZE)


def load_test_texts(dataset, author):
    if DATASETS[dataset]["type"] == "folders":
        return load_author_texts(os.path.join(DATASETS[dataset]["test"], author))

    return load_jsonl_by_author(DATASETS[dataset]["test"])[author]


def get_centroids(dataset, authors, embed_fn=embed_episode, cache_name="centroids.pt"):
    cache_path = os.path.join(config.CACHE_DIR, cache_name)
    if os.path.exists(cache_path):
        return torch.load(cache_path)
    centroids = {}
    for author in authors:
        print("calculating centroid for", author)
        support_texts = load_train_texts(dataset, author)
        centroids[author] = embed_fn(support_texts)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(centroids, cache_path)
    return centroids
