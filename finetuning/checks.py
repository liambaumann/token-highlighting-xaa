# sanity checks for the fine-tuning building blocks, run from the repo root: python -m finetuning.checks

import os
import random
import torch
from model import model, embed_chunked
from data import DATASETS
from finetuning.splits import get_split, load_reuters_split_files, make_reuters_split_files
from finetuning.episodes import sample_batch, text_pieces, AUTHORS_PER_BATCH, PIECES_PER_EPISODE
from finetuning.loss import supcon_loss
from finetuning.validation import embed_chunk32


def check_reuters_split():
    files = load_reuters_split_files()
    assert files == load_reuters_split_files() == make_reuters_split_files(), "split not reproducible"
    test_dir = DATASETS["reuters"]["test"]
    for author, f in files.items():
        assert len(f["train"]) == 40 and len(f["val"]) == 10, author
        assert not set(f["train"]) & set(f["val"]), f"{author}: file in both train and val"
        assert sorted(f["train"] + f["val"]) == sorted(os.listdir(os.path.join(DATASETS["reuters"]["train"], author)))
        assert not set(f["train"] + f["val"]) & set(os.listdir(os.path.join(test_dir, author))), f"{author}: C50test file"
    split = get_split("reuters")
    assert split == get_split("reuters")
    print(f"reuters split ok: {len(files)} authors, 40 train + 10 val each, disjoint, none from C50test, reproducible")


def piece_sources(split, author):
    # piece (as tuple of ids) -> set of indices of the author's training texts it occurs in
    sources = {}
    for i, text in enumerate(split[author]["train"]):
        for ids in text_pieces(text)[0]:
            sources.setdefault(tuple(ids), set()).add(i)
    return sources


def check_batch(dataset, split):
    rng = random.Random(0)
    a, p = AUTHORS_PER_BATCH[dataset], PIECES_PER_EPISODE
    input_ids, attention_mask, labels = sample_batch(split, a, p, rng)
    assert input_ids.shape == attention_mask.shape == (2 * a, p, 32), input_ids.shape
    assert labels.shape == (2 * a,)
    assert len(set(labels.tolist())) == a and (labels[0::2] == labels[1::2]).all()

    authors = sorted(split)
    for k in range(a):
        sources = piece_sources(split, authors[labels[2 * k]])
        # a piece can occur in several texts (e.g. duplicate comments), so only pieces that
        # occur in exactly one text pin down which text an episode used
        texts = [
            {min(sources[tuple(ids)]) for ids in input_ids[2 * k + e].tolist() if len(sources[tuple(ids)]) == 1}
            for e in range(2)
        ]
        assert not texts[0] & texts[1], f"{dataset}: episodes of {authors[labels[2 * k]]} share a text"
    print(f"{dataset} batch ok: input_ids/attention_mask {tuple(input_ids.shape)}, labels {tuple(labels.shape)}, "
          f"episodes of each author share no text")
    return input_ids, attention_mask, labels


def check_loss(input_ids, attention_mask, labels):
    with torch.no_grad():
        embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
    loss = supcon_loss(embeddings, labels)
    assert torch.isfinite(loss)

    first = embeddings[0::2]
    first_labels = labels[0::2]
    identical = supcon_loss(torch.cat([first, first]), torch.cat([first_labels, first_labels]))
    perm = torch.roll(torch.arange(len(first)), 1)  # every author's second episode from another author
    shuffled = supcon_loss(torch.cat([first, first[perm]]), torch.cat([first_labels, first_labels]))
    assert identical < shuffled
    print(f"loss ok: random batch {loss:.4f}, identical episodes {identical:.4f} < shuffled {shuffled:.4f}")


def check_embedding(text):
    diff = (embed_chunk32(model, [text]) - embed_chunked([text])).abs().max().item()
    assert diff <= 1e-5, diff
    print(f"chunk32 embedding ok: max abs difference to model.embed_chunked {diff:.2e}")


if __name__ == "__main__":
    check_reuters_split()
    reuters = get_split("reuters")
    darkreddit = get_split("darkreddit")
    batch = check_batch("reuters", reuters)
    check_batch("darkreddit", darkreddit)
    check_loss(*batch)
    check_embedding(reuters[sorted(reuters)[0]]["val"][0])
