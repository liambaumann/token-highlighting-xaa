# fine-tunes a fresh LUAR-MUD with the supervised contrastive loss on episode embeddings,
# validating after every epoch and keeping the model with the best validation MAP

import os
import json
import time
import random
import torch
from transformers import AutoModel
from model import MODELS
from finetuning.splits import get_split
from finetuning.episodes import sample_batch, AUTHORS_PER_BATCH
from finetuning.loss import supcon_loss
from finetuning.validation import validate

device = "cuda" if torch.cuda.is_available() else "cpu"

WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# LUAR-MUD's backbone is paraphrase-distilroberta-base-v1, which has 6 encoder layers (not 12)
NUM_ENCODER_LAYERS = 6

# settings that may change when a run is resumed; all others must match the checkpoint
RESUMABLE_SETTINGS = {"max_epochs", "patience"}


def run_name(settings):
    layers = "all" if settings["train_layers"] == NUM_ENCODER_LAYERS else f"top{settings['train_layers']}"
    name = f"{layers}_lr{settings['lr']:g}"
    # non-default authors per batch get their own suffix, so they never overwrite a default run
    if settings["authors_per_batch"] != AUTHORS_PER_BATCH[settings["dataset"]]:
        name += f"_apb{settings['authors_per_batch']}"
    return name + "_smoke" if settings["smoke_test"] else name


def run_paths(settings):
    name = run_name(settings)
    model_dir = os.path.join("models", "finetuned", settings["dataset"], name)
    return {
        "checkpoint": os.path.join(model_dir, "checkpoint.pt"),
        "best_model": os.path.join(model_dir, "best", "model.pt"),
        "best_info": os.path.join(model_dir, "best", "info.json"),
        "history": os.path.join("results", "finetuning", settings["dataset"], name, "history.json"),
    }


def load_luar():
    # a fresh LUAR-MUD, independent of the global model in model.py
    return AutoModel.from_pretrained(MODELS["luar-mud"], trust_remote_code=True)


def freeze(model, train_layers):
    # only the last train_layers encoder layers, attn_fn and linear stay trainable;
    # embeddings, lower layers (and the unused pooler) are frozen
    layers = model.transformer.encoder.layer
    assert len(layers) == NUM_ENCODER_LAYERS, f"expected {NUM_ENCODER_LAYERS} encoder layers, model has {len(layers)}"
    if not 0 <= train_layers <= len(layers):
        raise SystemExit(f"--train-layers must be between 0 and {len(layers)} (LUAR-MUD has {len(layers)} "
                         f"encoder layers; {len(layers)} trains all but the embeddings), got {train_layers}")
    model.requires_grad_(False)
    for layer in layers[len(layers) - train_layers:]:
        layer.requires_grad_(True)
    model.attn_fn.requires_grad_(True)
    model.linear.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


def training_step(model, optimizer, trainable, split, settings, sampler_rng):
    # loss and clipped gradients for one batch; the caller decides whether to optimizer.step()
    input_ids, attention_mask, labels = sample_batch(
        split, settings["authors_per_batch"], settings["pieces_per_episode"], sampler_rng
    )
    embeddings = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    loss = supcon_loss(embeddings, labels.to(device), settings["temperature"])
    if not torch.isfinite(loss):
        raise RuntimeError(f"training loss is not finite: {loss.item()}")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP)
    return loss.item()


def random_states(sampler_rng):
    states = {"python": random.getstate(), "torch": torch.get_rng_state(), "sampler": sampler_rng.getstate()}
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_random_states(states, sampler_rng):
    random.setstate(states["python"])
    torch.set_rng_state(states["torch"])
    sampler_rng.setstate(states["sampler"])
    if torch.cuda.is_available() and "cuda" in states:
        torch.cuda.set_rng_state_all(states["cuda"])


def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_checkpoint(paths, model, optimizer, epoch, best, history, settings, sampler_rng):
    os.makedirs(os.path.dirname(paths["checkpoint"]), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best": best,
        "random_states": random_states(sampler_rng),
        "history": history,
        "settings": settings,
    }, paths["checkpoint"])


def save_best(paths, model, settings, best):
    os.makedirs(os.path.dirname(paths["best_model"]), exist_ok=True)
    torch.save(model.state_dict(), paths["best_model"])
    save_json(paths["best_info"], {"run_name": run_name(settings), "settings": settings, **best})


def end_of_epoch(paths, model, optimizer, split, settings, sampler_rng, history, best, epoch, train_loss, epoch_start, run_start):
    # validate, update best model, history and checkpoint; returns the updated best
    top1, mAP = validate(model, settings["dataset"], split, settings["max_val_per_author"])
    if best is None or mAP > best["val_map"]:
        best = {"best_epoch": epoch, "val_top1": top1, "val_map": mAP}
        save_best(paths, model, settings, best)

    epoch_time = time.time() - epoch_start
    history["epochs"].append({
        "epoch": epoch, "train_loss": train_loss, "val_top1": top1, "val_map": mAP, "epoch_time": epoch_time,
    })
    save_json(paths["history"], history)
    save_checkpoint(paths, model, optimizer, epoch, best, history, settings, sampler_rng)

    loss_text = "-" if train_loss is None else f"{train_loss:.4f}"
    print(f"epoch {epoch}: train loss {loss_text}, val top-1 {top1:.2%}, val MAP {mAP:.2%}, "
          f"epoch time {epoch_time:.0f}s, total time {time.time() - run_start:.0f}s", flush=True)
    return best


def early_checks(paths, model, optimizer, trainable, split, settings, sampler_rng, history, run_start):
    # epoch 0: untrained model's validation, recorded in the history and saved as the first best
    best = end_of_epoch(paths, model, optimizer, split, settings, sampler_rng, history, None, 0, None, time.time(), run_start)

    # one training step on the first real batch, without updating the weights and without
    # consuming any random state, so the run itself is unaffected
    check_rng = random.Random()
    check_rng.setstate(sampler_rng.getstate())
    step_start = time.time()
    with torch.random.fork_rng():
        try:
            loss = training_step(model, optimizer, trainable, split, settings, check_rng)
        except RuntimeError as e:
            raise SystemExit(f"early check failed: training step: {e}")
    optimizer.zero_grad()
    print(f"check: training step ok, loss {loss:.4f}, step time {time.time() - step_start:.1f}s", flush=True)

    # the epoch-0 checkpoint must load back into a fresh model with identical weights
    fresh = load_luar()
    fresh.load_state_dict(torch.load(paths["checkpoint"], map_location="cpu")["model"])
    fresh_state = fresh.state_dict()
    mismatched = [k for k, v in model.state_dict().items() if not torch.equal(v.cpu(), fresh_state[k])]
    if mismatched:
        raise SystemExit(f"early check failed: checkpoint weights differ after loading: {mismatched[:5]}")
    del fresh, fresh_state
    print("check: checkpoint reloads into a fresh model with identical weights", flush=True)
    return best


def train(settings, resume=False):
    paths = run_paths(settings)
    print(f"run {run_name(settings)} on {device}", flush=True)
    print(json.dumps(settings), flush=True)

    random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    sampler_rng = random.Random(settings["seed"])

    model = load_luar().to(device)
    model.train()
    trainable = freeze(model, settings["train_layers"])
    num_trainable = sum(p.numel() for p in trainable)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"trainable parameters: {num_trainable:,} of {num_total:,} ({num_trainable / num_total:.1%})", flush=True)

    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=WEIGHT_DECAY)
    split = get_split(settings["dataset"])
    run_start = time.time()

    if resume:
        if not os.path.exists(paths["checkpoint"]):
            raise SystemExit(f"--resume: no checkpoint at {paths['checkpoint']}")
        checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
        changed = [k for k in settings if k not in RESUMABLE_SETTINGS and checkpoint["settings"].get(k) != settings[k]]
        if changed:
            raise SystemExit(f"--resume: settings differ from the checkpoint: "
                             + ", ".join(f"{k} {checkpoint['settings'].get(k)} -> {settings[k]}" for k in changed))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        set_random_states(checkpoint["random_states"], sampler_rng)
        history = checkpoint["history"]
        history["settings"] = settings
        best = checkpoint["best"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"resumed from epoch {checkpoint['epoch']} (best MAP {best['val_map']:.2%} at epoch {best['best_epoch']})", flush=True)
    else:
        history = {"run_name": run_name(settings), "settings": settings, "epochs": []}
        best = early_checks(paths, model, optimizer, trainable, split, settings, sampler_rng, history, run_start)
        start_epoch = 1

    for epoch in range(start_epoch, settings["max_epochs"] + 1):
        epochs_without_improvement = history["epochs"][-1]["epoch"] - best["best_epoch"]
        if epochs_without_improvement >= settings["patience"]:
            print(f"early stopping: no improvement in validation MAP for {epochs_without_improvement} epochs", flush=True)
            break

        model.train()
        epoch_start = time.time()
        losses = []
        for _ in range(settings["steps_per_epoch"]):
            losses.append(training_step(model, optimizer, trainable, split, settings, sampler_rng))
            optimizer.step()
        print(f"epoch {epoch}: training took {time.time() - epoch_start:.0f}s "
              f"({(time.time() - epoch_start) / settings['steps_per_epoch']:.1f}s per step)", flush=True)

        best = end_of_epoch(paths, model, optimizer, split, settings, sampler_rng, history, best,
                            epoch, sum(losses) / len(losses), epoch_start, run_start)
    else:
        print(f"reached max_epochs {settings['max_epochs']}", flush=True)

    print(f"best epoch {best['best_epoch']}: val top-1 {best['val_top1']:.2%}, val MAP {best['val_map']:.2%}", flush=True)
    return best
