# train/validation splits for fine-tuning; the test sets are never read here
# reuters: 40 of each author's 50 C50train documents for training, 10 held back for validation
# darkreddit: the whole train file for training, the val file for validation

import os
import json
import random
from data import DATASETS, load_authors, load_jsonl_by_author

REUTERS_VAL_PER_AUTHOR = 10
REUTERS_SPLIT_SEED = 0
SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits")
REUTERS_SPLIT_PATH = os.path.join(SPLITS_DIR, "reuters.json")


def make_reuters_split_files():
    # author -> {"train": [file names], "val": [file names]}, drawn from C50train only
    train_dir = DATASETS["reuters"]["train"]
    rng = random.Random(REUTERS_SPLIT_SEED)
    split_files = {}
    for author in load_authors("reuters"):
        files = sorted(os.listdir(os.path.join(train_dir, author)))
        val_files = set(rng.sample(files, REUTERS_VAL_PER_AUTHOR))
        split_files[author] = {
            "train": [f for f in files if f not in val_files],
            "val": [f for f in files if f in val_files],
        }
    return split_files


def load_reuters_split_files():
    # created on first use, afterwards always read back from the json so every run uses the same split
    if not os.path.exists(REUTERS_SPLIT_PATH):
        os.makedirs(SPLITS_DIR, exist_ok=True)
        with open(REUTERS_SPLIT_PATH, "w") as f:
            json.dump(make_reuters_split_files(), f, indent=2)
    with open(REUTERS_SPLIT_PATH) as f:
        return json.load(f)


def read_texts(folder, filenames):
    texts = []
    for filename in filenames:
        with open(os.path.join(folder, filename), encoding="utf-8") as f:
            texts.append(f.read())
    return texts


def get_split(dataset):
    # author -> {"train": [texts], "val": [texts]}
    if dataset == "reuters":
        train_dir = DATASETS["reuters"]["train"]
        split = {}
        for author, files in load_reuters_split_files().items():
            folder = os.path.join(train_dir, author)
            split[author] = {"train": read_texts(folder, files["train"]), "val": read_texts(folder, files["val"])}
        return split

    if dataset == "darkreddit":
        train = load_jsonl_by_author(DATASETS["darkreddit"]["train"])
        val = load_jsonl_by_author(DATASETS["darkreddit"]["val"])
        return {author: {"train": train[author], "val": val[author]} for author in sorted(train)}

    raise ValueError(f"unknown dataset: {dataset!r}")
