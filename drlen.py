import json, os, numpy as np
from transformers import AutoTokenizer

DR_DIR = "./data/darkreddit_authorship_attribution_anon"
DR_SPLITS = ["train", "val", "test"]
DR_FIELD = "comment"              # key holding the post text

REUTERS_DIR = "./data/reuter_50_50"
REUTERS_SPLITS = ["C50train", "C50test"]

tok = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)


def load_darkreddit(split):
    path = f"{DR_DIR}/darkreddit_authorship_attribution_{split}_anon.jsonl"
    with open(path, encoding="utf-8") as f:
        return [json.loads(l)[DR_FIELD] for l in f]


def load_reuters(split):
    split_dir = os.path.join(REUTERS_DIR, split)
    texts = []
    for author in sorted(os.listdir(split_dir)):
        author_dir = os.path.join(split_dir, author)
        if not os.path.isdir(author_dir):
            continue
        for filename in sorted(os.listdir(author_dir)):
            with open(os.path.join(author_dir, filename), encoding="utf-8") as f:
                texts.append(f.read())
    return texts


def report(name, texts):
    lens = np.array([len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts])

    print(f"\n== {name} ==")
    print("docs:", len(lens))
    print(f"mean: {lens.mean():.1f}  min: {lens.min()}  max: {lens.max()}")
    print("percentiles 10/25/50/75/90:", np.percentile(lens, [10, 25, 50, 75, 90]))
    for n in [16, 32, 64, 128, 256, 512]:
        print(f">= {n} tokens: {(lens >= n).mean():.1%}")

    words = [t.split() for t in texts]
    word_counts = np.array([len(w) for w in words])
    word_chars = np.array([len(w) for ws in words for w in ws])
    print(f"words/post  mean: {word_counts.mean():.1f}  median: {np.median(word_counts):.0f}")
    print(f"chars/word  mean: {word_chars.mean():.2f}  median: {np.median(word_chars):.0f}")
    print(f"tokens/word: {lens.sum() / word_counts.sum():.2f}")


for split in DR_SPLITS:
    report(f"darkreddit {split}", load_darkreddit(split))

for split in REUTERS_SPLITS:
    report(f"reuters {split}", load_reuters(split))
