# author retrieval on the validation split: one chunk32 centroid per author from training
# texts, every validation text embedded as a chunk32 query
# run from the repo root for the baseline: python -m finetuning.validation reuters

import sys
import time
import torch
from model import chunk_pieces
from data import load_train_texts
from retrieval import top1_accuracy, mean_avg_precision
from finetuning.splits import get_split


def embed_chunk32(model, texts):
    # same as model.embed_chunked, but for the given model: all pieces of all texts form one episode
    all_ids = []
    all_mask = []
    for text in texts:
        pieces_ids, pieces_mask = chunk_pieces(text)
        all_ids.extend(pieces_ids)
        all_mask.extend(pieces_mask)

    device = next(model.parameters()).device
    input_ids = torch.tensor(all_ids).reshape(1, len(all_ids), 32).to(device)
    attention_mask = torch.tensor(all_mask).reshape(1, len(all_mask), 32).to(device)
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).cpu()


def centroid_texts(dataset, split, author):
    # darkreddit uses the same 50-comment sample as retrieval.py, so the numbers stay comparable
    if dataset == "darkreddit":
        return load_train_texts(dataset, author)
    return split[author]["train"]


def validate(model, dataset, split, max_val_per_author=None):
    # returns (top-1 accuracy, MAP); the model is put in eval mode and restored afterwards
    # max_val_per_author: only use the first n validation texts of each author (None: all)
    was_training = model.training
    model.eval()

    authors = sorted(split)
    centroids = torch.stack([embed_chunk32(model, centroid_texts(dataset, split, a)) for a in authors])

    sims = []
    true_labels = []
    for i, author in enumerate(authors):
        for text in split[author]["val"][:max_val_per_author]:
            query = embed_chunk32(model, [text])
            sims.append(torch.nn.functional.cosine_similarity(query.unsqueeze(0), centroids, dim=1))
            true_labels.append(i)
    sims = torch.stack(sims)
    true_labels = torch.tensor(true_labels)

    model.train(was_training)

    correct, total = top1_accuracy(sims, true_labels)
    return correct / total, mean_avg_precision(sims, true_labels)


if __name__ == "__main__":
    from model import model

    for dataset in sys.argv[1:] or ["reuters", "darkreddit"]:
        start = time.time()
        top1, mAP = validate(model, dataset, get_split(dataset))
        print(f"{dataset}: top-1 {top1:.2%}, MAP {mAP:.2%} ({time.time() - start:.0f}s)")
