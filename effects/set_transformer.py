# effects: a Set Transformer as a matching method. It maps the signature units that occur in a
# comment onto the author centroid in LUAR's space; compared with TF-SVM and Δ-sum (effects.matching)
# on the same signatures. Runs locally on CPU; nothing is embedded.
# Run from the repo root: python -m effects.set_transformer   (--unit: unit type, default word;
# --n: n-gram size, only 1 for now)
#
# Data roles: val gives the signatures and the Δ-vectors (the latest scoring and deltas files, as in
# matching); train (all train comments minus a 10 % holdout per author, seed 0) trains the model; the
# "train holdout" only stops training and chooses the width; test (all test comments, read through
# matching.load_test) is read once, after every model is trained and every width chosen.
# Targets: the cached LUAR author centroids (config.CACHE_DIR/centroids.pt, the ones the retrieval
# ceiling uses), loaded, not recomputed.
#
# Pooled units: the union of all authors' top-k units of the signature type. A comment's set has one
# element per pooled unit that occurs in it (units as matching.unit_counts: first 510 content
# tokens): the unit's vector at model width (--unit-vectors learned: an embedding table;
# background-delta: the fixed background Δ = raw - corrected Δ, through a learned linear layer)
# concatenated with count / the comment's number of units, through one linear layer to the width.
# A comment with none of the units gets one learned "empty" element instead.
# Model: 2 self-attention blocks (4 heads, feedforward 4 × width), attention pooling with one learned
# seed vector, one linear layer to 512. No positional information: the output ignores element order.
# Training: cross-entropy over cos(output, centroid) / 0.07 for the 10 authors; AdamW (lr 1e-3, weight
# decay 0.01), batch 64, at most 100 epochs, stop after 10 epochs without a better holdout top-1, keep
# the best holdout epoch's weights. Widths 32, 64, 128 per run; the best holdout top-1 is evaluated on
# test. Runs: signature types score1 and frequency × both unit vectors, k = 10.
#
# Writes results/effects/{dataset}/{perturbation}/{model_mode}/{unit}/n1/set_transformer_{timestamp}.json
# and the chosen models to models/effects/set_transformer/{dataset}/{perturbation}/{model_mode}/{unit}/n1/;
# the page (outputs/{dataset}/{perturbation}/{model_mode}/effects_set_transformer_{unit}_{timestamp}.html)
# is built from the result JSON by effects.set_transformer_page.

import os
import json
import copy
import time
import random
import argparse
from datetime import datetime
import torch
from torch import nn
import config
from data import DATASETS, load_jsonl_by_author
from metrics import top1_accuracy
from runs import load_runs
from .units import UNIT_TYPES, unit_key
from .unit_scores import unit_dir
from .matching import (
    MATCHING_PREFIX, load_scoring, frequency_ranking, signature_sources, top_signatures, load_train, load_test,
    feature_order,
)
from .set_transformer_page import save_result_page

RESULT_PREFIX = "set_transformer_"
MODELS_ROOT = "models/effects/set_transformer"
# the plain-mode centroids of data.get_centroids, as the retrieval ceiling uses them
CENTROIDS_FILE = os.path.join(config.CACHE_DIR, "centroids.pt")

NGRAM_SIZES = [1]
K = 10
SIGNATURE_TYPES = [("score1", "score 1"), ("frequency", "frequency")]
UNIT_VECTORS = ["learned", "background-delta"]
WIDTHS = [32, 64, 128]

NUM_BLOCKS = 2
NUM_HEADS = 4
FEEDFORWARD_FACTOR = 4
OUTPUT_DIM = 512
DROPOUT = 0.0
TEMPERATURE = 0.07
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10
SEED = 0
HOLDOUT_FRACTION = 0.1
HOLDOUT_SEED = 0
EVAL_BATCH_SIZE = 512
PERMUTATION_TOLERANCE = 1e-5
PERMUTATION_CHECK_COMMENTS = 256
# the matching methods compared at k = K
MATCHING_METHODS = ["tf_svm", "delta_sum"]


# ---------- inputs ----------

def load_centroids(authors):
    # (authors × 512) in author order, from the cached file
    if config.MODE != "plain":
        raise SystemExit(f"effects.set_transformer uses the plain-mode centroids ({CENTROIDS_FILE}), not MODE {config.MODE!r}")
    if not os.path.exists(CENTROIDS_FILE):
        raise SystemExit(f"no {CENTROIDS_FILE} - run retrieval first")
    centroids = torch.load(CENTROIDS_FILE)
    if set(centroids) != set(authors):
        raise SystemExit(f"{CENTROIDS_FILE} has authors {sorted(centroids)}, the scoring {authors}")
    return torch.stack([centroids[author] for author in authors]).float()


def load_matching(directory, scoring):
    # the latest matching result; it must use the same scoring file, else its numbers are on other signatures
    runs = load_runs(directory, MATCHING_PREFIX)
    if not runs:
        raise SystemExit(f"no {MATCHING_PREFIX}*.json files in {directory} - run effects.matching first")
    matching = runs[-1]
    if matching["scoring_file"] != scoring["file"]:
        raise SystemExit(f"{matching['file']} uses {matching['scoring_file']}, the latest scoring is {scoring['file']} - "
                         f"run effects.matching again")
    return matching


def background_deltas(scoring, directory, pooled):
    # (pooled units × 512): per unit raw - corrected Δ = its background Δ; identical for every author
    # who has the unit up to float32 rounding (the largest difference is returned), the first author's taken
    deltas = torch.load(os.path.join(directory, scoring["deltas_file"]))
    background = deltas["raw"].double() - deltas["corrected"].double()
    per_unit = {}
    for author in scoring["authors"]:
        for entry in scoring["scores"][author]:
            per_unit.setdefault(unit_key(entry["unit"]), []).append(background[entry["delta_row"]])
    mismatch = max((torch.stack(rows) - rows[0]).abs().max().item() for rows in (per_unit[u] for u in pooled))
    return torch.stack([per_unit[u][0] for u in pooled]).float(), mismatch


def holdout_split(labels, num_authors):
    # per author HOLDOUT_FRACTION of its train comments (rounded), one random.Random(HOLDOUT_SEED) per author
    train_ids, holdout_ids = [], []
    for a in range(num_authors):
        ids = [i for i, label in enumerate(labels) if label == a]
        held = set(random.Random(HOLDOUT_SEED).sample(ids, round(HOLDOUT_FRACTION * len(ids))))
        train_ids += [i for i in ids if i not in held]
        holdout_ids += sorted(held)
    return train_ids, holdout_ids


def pooled_units(signatures):
    return sorted({unit_key(entry["unit"]) for entries in signatures.values() for entry in entries}, key=feature_order)


def comment_sets(comment_counts, pooled):
    # per comment (pooled indices, count / number of units), in pooled order; a comment with none of the
    # units gets ([len(pooled)], [0.0]), the index of the empty element
    index = {unit: j for j, unit in enumerate(pooled)}
    sets = []
    for counts, num_units in comment_counts:
        items = sorted((index[unit], count / num_units) for unit, count in counts.items() if unit in index)
        if not items:
            items = [(len(pooled), 0.0)]
        sets.append((torch.tensor([j for j, _ in items]), torch.tensor([v for _, v in items], dtype=torch.float32)))
    return sets


def count_empty(sets, ids, num_units):
    return sum(1 for i in ids if sets[i][0][0].item() == num_units)


def collate(sets, ids):
    # padded (indices, values, padding mask) of the comments ids
    lengths = [len(sets[i][0]) for i in ids]
    length = max(lengths)
    indices = torch.zeros(len(ids), length, dtype=torch.long)
    values = torch.zeros(len(ids), length)
    padding = torch.ones(len(ids), length, dtype=torch.bool)
    for row, (i, n) in enumerate(zip(ids, lengths)):
        indices[row, :n], values[row, :n], padding[row, :n] = sets[i][0], sets[i][1], False
    return indices, values, padding


# ---------- model ----------

class SetTransformer(nn.Module):
    def __init__(self, num_units, width, unit_vectors, background=None):
        super().__init__()
        self.num_units = num_units
        self.unit_vectors = unit_vectors
        if unit_vectors == "learned":
            self.unit_embedding = nn.Embedding(num_units, width)
        else:
            # fixed background Δ per unit, projected to the width
            self.register_buffer("background", background)
            self.background_projection = nn.Linear(background.shape[1], width)
        # unit vector and count share -> one element
        self.input = nn.Linear(width + 1, width)
        self.empty = nn.Parameter(torch.randn(width))
        block = nn.TransformerEncoderLayer(width, NUM_HEADS, FEEDFORWARD_FACTOR * width, dropout=DROPOUT, batch_first=True)
        self.blocks = nn.TransformerEncoder(block, NUM_BLOCKS, enable_nested_tensor=False)
        self.seed = nn.Parameter(torch.randn(1, 1, width))
        self.pool = nn.MultiheadAttention(width, NUM_HEADS, dropout=DROPOUT, batch_first=True)
        self.output = nn.Linear(width, OUTPUT_DIM)

    def forward(self, indices, values, padding):
        empty = indices == self.num_units
        safe = indices.clamp(max=self.num_units - 1)
        if self.unit_vectors == "learned":
            vectors = self.unit_embedding(safe)
        else:
            vectors = self.background_projection(self.background[safe])
        elements = self.input(torch.cat([vectors, values.unsqueeze(-1)], dim=-1))
        elements = torch.where(empty.unsqueeze(-1), self.empty.expand_as(elements), elements)
        hidden = self.blocks(elements, src_key_padding_mask=padding)
        pooled, _ = self.pool(self.seed.expand(len(hidden), -1, -1), hidden, hidden, key_padding_mask=padding, need_weights=False)
        return self.output(pooled.squeeze(1))


def predict(model, sets, ids):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(*collate(sets, ids[b:b + EVAL_BATCH_SIZE])) for b in range(0, len(ids), EVAL_BATCH_SIZE)])


def cosine_sims(outputs, centroids):
    # (comments × authors) cosine similarity to every centroid
    return nn.functional.normalize(outputs, dim=1) @ nn.functional.normalize(centroids, dim=1).T


def evaluate(model, sets, labels, ids, centroids):
    # top-1 (nearest centroid by cosine) and the training loss on the comments ids
    sims = cosine_sims(predict(model, sets, ids), centroids)
    targets = labels[ids]
    correct, total = top1_accuracy(sims, targets)
    loss = nn.functional.cross_entropy(sims / TEMPERATURE, targets).item()
    return {"top1": correct / total, "correct": correct, "total": total, "loss": loss}


def permutation_check(model, sets, ids):
    # largest output difference between the sets and their elements shuffled, over the first
    # PERMUTATION_CHECK_COMMENTS comments of ids with at least two elements
    chosen = [i for i in ids if len(sets[i][0]) >= 2][:PERMUTATION_CHECK_COMMENTS]
    generator = torch.Generator().manual_seed(SEED)
    shuffled = []
    for i in chosen:
        order = torch.randperm(len(sets[i][0]), generator=generator)
        shuffled.append((sets[i][0][order], sets[i][1][order]))
    original = predict(model, sets, chosen)
    permuted = predict(model, shuffled, list(range(len(shuffled))))
    return {"max_abs_diff": (original - permuted).abs().max().item(), "num_comments": len(chosen)}


def train_one(sets, labels, train_ids, holdout_ids, centroids, num_units, width, unit_vectors, background):
    # one training run; returns the model with its best holdout epoch's weights and the run's record
    start = time.time()
    torch.manual_seed(SEED)
    model = SetTransformer(num_units, width, unit_vectors, background)
    permutation_initial = permutation_check(model, sets, holdout_ids)
    if permutation_initial["max_abs_diff"] >= PERMUTATION_TOLERANCE:
        raise SystemExit(f"untrained model is not permutation invariant: {permutation_initial}")
    initial = {"train": evaluate(model, sets, labels, train_ids, centroids),
               "holdout": evaluate(model, sets, labels, holdout_ids, centroids)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(SEED)
    targets = nn.functional.normalize(centroids, dim=1)

    epochs = []
    best_top1, best_epoch, best_state, since_best = -1.0, 0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_ids), generator=generator).tolist()
        total_loss = 0.0
        for b in range(0, len(order), BATCH_SIZE):
            ids = [train_ids[j] for j in order[b:b + BATCH_SIZE]]
            outputs = model(*collate(sets, ids))
            logits = nn.functional.normalize(outputs, dim=1) @ targets.T / TEMPERATURE
            loss = nn.functional.cross_entropy(logits, labels[ids])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ids)
        train = evaluate(model, sets, labels, train_ids, centroids)
        holdout = evaluate(model, sets, labels, holdout_ids, centroids)
        epochs.append({"epoch": epoch, "train_loss": total_loss / len(train_ids), "train_top1": train["top1"],
                       "holdout_top1": holdout["top1"], "holdout_loss": holdout["loss"]})
        if holdout["top1"] > best_top1:
            best_top1, best_epoch, best_state, since_best = holdout["top1"], epoch, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
        if since_best >= PATIENCE:
            break
    model.load_state_dict(best_state)
    record = {
        "width": width,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "initial": initial,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_holdout_top1": best_top1,
        "stopped_after_epoch": epochs[-1]["epoch"],
        "permutation_check_initial": permutation_initial,
        "permutation_check_best": permutation_check(model, sets, holdout_ids),
        "seconds": time.time() - start,
    }
    return model, record


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Set Transformer matching on the effects signatures")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="word")
    parser.add_argument("--n", type=int, choices=NGRAM_SIZES, default=1)
    return parser.parse_args()


def main():
    script_start = time.time()
    args = parse_args()
    unit, n = args.unit, args.n
    created_at = datetime.now()
    stamp = f"{created_at:%Y%m%d-%H%M%S}"
    directory = unit_dir(unit, n)

    scoring, _ = load_scoring(directory)
    authors = scoring["authors"]
    matching = load_matching(directory, scoring)
    centroids = load_centroids(authors)
    print(f"scoring {scoring['file']}, matching {matching['file']}, centroids {CENTROIDS_FILE}")

    # training data: val (signatures, Δ) and train only
    frequency = frequency_ranking(directory, scoring, authors, unit, n)
    sources = signature_sources(scoring, frequency)
    train_counts, train_label_list = load_train(authors, unit, n)
    labels = torch.tensor(train_label_list)
    train_ids, holdout_ids = holdout_split(train_label_list, len(authors))
    assert not set(train_ids) & set(holdout_ids) and len(train_ids) + len(holdout_ids) == len(train_label_list)
    print(f"train {len(train_ids)}, holdout {len(holdout_ids)}")

    runs, models, inputs_per_type = {}, {}, {}
    for key, label in SIGNATURE_TYPES:
        signatures = top_signatures(sources[key], authors, key, K)
        if key == "frequency":
            # the same ranking as the matching run's (its saved frequency ranking)
            saved = top_signatures(matching["frequency_ranking"], authors, key, K)
            if any([unit_key(e["unit"]) for e in signatures[a]] != [unit_key(e["unit"]) for e in saved[a]] for a in authors):
                raise SystemExit(f"frequency signatures differ from those in {matching['file']}")
        pooled = pooled_units(signatures)
        background, mismatch = background_deltas(scoring, directory, pooled)
        sets = comment_sets(train_counts, pooled)
        inputs_per_type[key] = {"pooled": pooled, "background": background, "background_mismatch": mismatch,
                                "signatures": {a: [unit_key(e["unit"]) for e in signatures[a]] for a in authors},
                                "empty_train": count_empty(sets, train_ids, len(pooled)),
                                "empty_holdout": count_empty(sets, holdout_ids, len(pooled))}
        print(f"{label}: {len(pooled)} pooled units, empty sets train {inputs_per_type[key]['empty_train']}, "
              f"holdout {inputs_per_type[key]['empty_holdout']}")
        runs[key], models[key] = {}, {}
        for vectors in UNIT_VECTORS:
            records, trained = [], {}
            for width in WIDTHS:
                model, record = train_one(sets, labels, train_ids, holdout_ids, centroids, len(pooled), width, vectors,
                                          background if vectors == "background-delta" else None)
                records.append(record)
                trained[width] = model
                print(f"  {vectors}, width {width}: before {record['initial']['holdout']['top1']:.2%}, best holdout "
                      f"{record['best_holdout_top1']:.2%} at epoch {record['best_epoch']}/{record['stopped_after_epoch']}, "
                      f"{record['seconds']:.0f}s")
            # best holdout top-1, ties to the smaller width
            chosen = max(records, key=lambda r: (r["best_holdout_top1"], -r["width"]))
            runs[key][vectors] = {"widths": records, "chosen_width": chosen["width"], "chosen_holdout_top1": chosen["best_holdout_top1"]}
            models[key][vectors] = trained[chosen["width"]]

    # evaluation: the only place test is read, after every model is trained and every width chosen
    test_start = time.time()
    _, test_counts, test_labels, test_files = load_test(directory, scoring["cache_series"], authors, unit, n)
    test_ids = list(range(len(test_labels)))
    empty_sets = {}
    for key, label in SIGNATURE_TYPES:
        pooled = inputs_per_type[key]["pooled"]
        test_sets = comment_sets(test_counts, pooled)
        empty_sets[key] = {"train": inputs_per_type[key]["empty_train"], "holdout": inputs_per_type[key]["empty_holdout"],
                           "test": count_empty(test_sets, test_ids, len(pooled))}
        for vectors in UNIT_VECTORS:
            sims = cosine_sims(predict(models[key][vectors], test_sets, test_ids), centroids)
            correct, total = top1_accuracy(sims, test_labels)
            runs[key][vectors]["test"] = {"top1": correct / total, "correct": correct, "total": total}

    # data check, after the evaluation: test comments whose text is also a train comment
    train_texts = {text for texts in load_jsonl_by_author(DATASETS[config.DATASET]["train"]).values() for text in texts}
    test_texts = [text for texts in load_jsonl_by_author(DATASETS[config.DATASET]["test"]).values() for text in texts]
    test_in_train = sum(1 for text in test_texts if text in train_texts)

    # the chosen models
    model_dir = os.path.join(MODELS_ROOT, config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME, unit, f"n{n}")
    os.makedirs(model_dir, exist_ok=True)
    model_files = {}
    for key, _ in SIGNATURE_TYPES:
        model_files[key] = {}
        for vectors in UNIT_VECTORS:
            width = runs[key][vectors]["chosen_width"]
            path = os.path.join(model_dir, f"{stamp}_{key}_{vectors}_w{width}.pt")
            torch.save({"state_dict": models[key][vectors].state_dict(), "width": width, "unit_vectors": vectors,
                        "signature_type": key, "k": K, "pooled_units": inputs_per_type[key]["pooled"], "authors": authors,
                        "result_file": f"{RESULT_PREFIX}{stamp}.json"}, path)
            model_files[key][vectors] = path

    matching_k10 = {key: {method: matching["results"][method][key][str(K)]["top1"] for method in MATCHING_METHODS}
                    for key, _ in SIGNATURE_TYPES}
    result = {
        "experiment": "effects_set_transformer",
        "created_at": created_at.isoformat(timespec="seconds"),
        "file": f"{RESULT_PREFIX}{stamp}.json",
        "dataset": scoring["dataset"],
        "model": scoring["model"],
        "mode": scoring["mode"],
        "perturbation": scoring["perturbation"],
        "unit": unit,
        "ngram": n,
        "authors": authors,
        "inputs": {
            "scoring_file": scoring["file"],
            "deltas_file": scoring["deltas_file"],
            "matching_file": matching["file"],
            "centroids_file": CENTROIDS_FILE,
            "cache_series": scoring["cache_series"],
            "cache_val_files": scoring["cache_files"],
            "cache_test_files": test_files,
            "train_data": DATASETS[config.DATASET]["train"],
            "test_data": DATASETS[config.DATASET]["test"],
        },
        "matching_file": matching["file"],
        "settings": {
            "k": K,
            "signature_types": dict(SIGNATURE_TYPES),
            "unit_vectors": {
                "learned": "nn.Embedding(pooled units, width)",
                "background-delta": "raw - corrected Δ of the unit from the deltas file (fixed buffer), nn.Linear(512, width)",
            },
            "element": "concat(unit vector at width, count in comment / comment's number of units) -> nn.Linear(width + 1, width); "
                       "a comment with none of the pooled units: one learned empty vector (width) as its only element",
            "units": "matching.unit_counts (effects.cache tokenisation, first 510 content tokens)",
            "pooled_units": "union of all authors' top-k units of the signature type",
            "widths": WIDTHS,
            "blocks": NUM_BLOCKS, "heads": NUM_HEADS, "feedforward": f"{FEEDFORWARD_FACTOR} x width",
            "block_type": "nn.TransformerEncoderLayer (post-norm, ReLU), key padding mask, no positional information",
            "pooling": "nn.MultiheadAttention with one learned seed vector as query",
            "output_dim": OUTPUT_DIM,
            "dropout": DROPOUT,
            "loss": f"cross-entropy over cos(output, centroid) / {TEMPERATURE} for all authors",
            "optimizer": {"name": "AdamW", "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY},
            "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "stopping": "stop when holdout top-1 has not improved (strictly) for patience epochs; weights of the best holdout epoch kept",
            "train_loss": "mean of the batch losses during the epoch; train top-1: eval pass after the epoch",
            "width_choice": "best holdout top-1, ties to the smaller width; only the chosen width evaluated on test",
            "seed": SEED, "seeding": "torch.manual_seed(seed) before each model; batch order from torch.Generator seeded with seed",
            "holdout": f"per author round({HOLDOUT_FRACTION} x its train comments), random.Random({HOLDOUT_SEED}).sample per author",
            "test_top1": "nearest centroid by cosine (metrics.top1_accuracy)",
            "scoring_settings": scoring["settings"],
            "cache_settings": scoring["cache_settings"],
        },
        "data": {"num_val_comments": sum(scoring["num_comments"].values()), "num_train_comments": len(train_ids),
                 "num_holdout_comments": len(holdout_ids), "num_test_comments": len(test_labels)},
        "checks": {
            "train_holdout_disjoint": True,
            "test_read_after_width_choice": True,
            "test_texts_also_in_train": test_in_train,
            "background_delta_max_mismatch_across_authors": {key: inputs_per_type[key]["background_mismatch"] for key, _ in SIGNATURE_TYPES},
            "permutation_tolerance": PERMUTATION_TOLERANCE,
        },
        "num_pooled_units": {key: len(inputs_per_type[key]["pooled"]) for key, _ in SIGNATURE_TYPES},
        "pooled_units": {key: inputs_per_type[key]["pooled"] for key, _ in SIGNATURE_TYPES},
        "signatures": {key: inputs_per_type[key]["signatures"] for key, _ in SIGNATURE_TYPES},
        "empty_sets": empty_sets,
        "matching_k10": matching_k10,
        "model_files": model_files,
        "runs": runs,
        "seconds": {"total": time.time() - script_start, "test_evaluation": time.time() - test_start},
    }

    result_path = os.path.join(directory, result["file"])
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"saved results to {result_path}")
    save_result_page(result, stamp)

    for key, label in SIGNATURE_TYPES:
        for vectors in UNIT_VECTORS:
            run = runs[key][vectors]
            print(f"{label}, {vectors}: width {run['chosen_width']}, holdout {run['chosen_holdout_top1']:.2%}, "
                  f"test {run['test']['top1']:.2%} (TF-SVM {matching_k10[key]['tf_svm']:.2%}, Δ-sum {matching_k10[key]['delta_sum']:.2%})")
    print(f"empty sets: {empty_sets}; test texts also in train: {test_in_train}; total {result['seconds']['total']:.0f}s")


if __name__ == "__main__":
    main()
