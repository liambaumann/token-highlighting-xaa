# entry point, run from the repo root:
# python -m finetuning.finetune --dataset reuters --train-layers 2 --lr 2e-5

import argparse
from finetuning.episodes import AUTHORS_PER_BATCH, PIECES_PER_EPISODE
from finetuning.loss import DEFAULT_TEMPERATURE
from finetuning.trainer import train

SMOKE_EPOCHS = 2
SMOKE_STEPS = 3
SMOKE_VAL_PER_AUTHOR = 2


def parse_args():
    parser = argparse.ArgumentParser(description="fine-tune LUAR-MUD with a supervised contrastive loss")
    parser.add_argument("--dataset", required=True, choices=["reuters", "darkreddit"])
    parser.add_argument("--train-layers", type=int, default=2, help="number of top encoder layers to train (LUAR-MUD has 6; 6 trains all but the embeddings)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-epochs", type=int, default=None, help=f"default 20 ({SMOKE_EPOCHS} with --smoke-test)")
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--authors-per-batch", type=int, default=None, help=f"default {AUTHORS_PER_BATCH}")
    parser.add_argument("--pieces-per-episode", type=int, default=PIECES_PER_EPISODE)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pt of the same run")
    parser.add_argument("--smoke-test", action="store_true",
                        help=f"{SMOKE_EPOCHS} epochs of {SMOKE_STEPS} steps, validation on {SMOKE_VAL_PER_AUTHOR} texts per author")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings = {
        "dataset": args.dataset,
        "train_layers": args.train_layers,
        "lr": args.lr,
        "max_epochs": args.max_epochs if args.max_epochs is not None else (SMOKE_EPOCHS if args.smoke_test else 20),
        "steps_per_epoch": SMOKE_STEPS if args.smoke_test else args.steps_per_epoch,
        "patience": args.patience,
        "authors_per_batch": args.authors_per_batch or AUTHORS_PER_BATCH[args.dataset],
        "pieces_per_episode": args.pieces_per_episode,
        "temperature": args.temperature,
        "seed": args.seed,
        "smoke_test": args.smoke_test,
        "max_val_per_author": SMOKE_VAL_PER_AUTHOR if args.smoke_test else None,
    }
    train(settings, resume=args.resume)
