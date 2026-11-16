# effects: zero-filled vs available-case background mean, side by side, as markdown tables.
# Run from the repo root: python -m effects.background_table --unit word --n 1
#
# Reads the latest matching_*.json of the unit type and n from both means' folders
# (unit_scores.result_dir: n{n}/ for zero-filled, n{n}/available-case/ for available-case) and
# stops if either is missing or its "mean" field doesn't name its folder's mean. Prints one table
# per method (TF-SVM, Δ-sum), top-1 on test in %: one row per signature type and mean, random as
# mean (min–max) over the seeds, columns k = 1 ... 50; below each table k = all, the number of test
# comments, the ceiling and both source files. Writes nothing; stdout only.

import os
import json
import argparse
from .units import UNIT_TYPES, NGRAM_SIZES, COMBINED
from .unit_scores import MEANS, result_dir

# as in effects.matching, which isn't imported because it loads the model
MATCHING_PREFIX = "matching_"
METHODS = [("tf_svm", "TF-SVM"), ("delta_sum", "Δ-sum")]
SCORES = [("score1", "score 1"), ("score1_d20", "score 1 (d = 20)"), ("score2", "score 2"), ("frequency", "frequency")]
RANDOM_LABEL = "random"
TABLE_KS = ["1", "2", "5", "10", "20", "50"]
ALL_K = "all"


def load_latest(unit, n, mean):
    # the latest matching file of the mean's folder, with its file name under "file"
    directory = result_dir(unit, n, mean)
    files = sorted(f for f in os.listdir(directory) if f.startswith(MATCHING_PREFIX) and f.endswith(".json")) \
        if os.path.isdir(directory) else []
    if not files:
        raise SystemExit(f"no {MATCHING_PREFIX}*.json in {directory} - run "
                         f"python -m effects.matching --unit {unit} --n {n} --mean {mean} first")
    with open(os.path.join(directory, files[-1])) as f:
        result = json.load(f)
    if result.get("mean") != mean:
        raise SystemExit(f"{files[-1]} in {directory} has mean {result.get('mean')!r}, expected {mean!r} - "
                         f"rerun python -m effects.matching --unit {unit} --n {n} --mean {mean}")
    result["file"] = files[-1]
    return result


def pct(rate):
    return f"{rate * 100:.1f}"


def random_cell(random_k):
    return f"{pct(random_k['mean'])} ({pct(random_k['min'])}–{pct(random_k['max'])})"


def cell(method_results, key, k):
    return random_cell(method_results[key][k]) if key == RANDOM_LABEL else pct(method_results[key][k]["top1"])


def same_or_both(values, fmt):
    # one value if both means agree, else each with its mean
    if values["zero-filled"] == values["available-case"]:
        return fmt(values["zero-filled"])
    return ", ".join(f"{fmt(value)} ({mean})" for mean, value in values.items())


def print_method(method, label, results):
    keys = [(key, name) for key, name in SCORES] + [(RANDOM_LABEL, RANDOM_LABEL)]
    print(f"### {label}\n")
    print("| signature | mean | " + " | ".join(f"k = {k}" for k in TABLE_KS) + " |")
    print("|---|---|" + "---:|" * len(TABLE_KS))
    for key, _ in keys:
        for mean in MEANS:
            method_results = results[mean]["results"][method]
            print(f"| {key} | {mean} | " + " | ".join(cell(method_results, key, k) for k in TABLE_KS) + " |")
    print()
    # k = all: every signature type is all cached units, so one rate per mean (per type if they ever differ)
    all_rates = {mean: {key: results[mean]["results"][method][key][ALL_K] for key, _ in keys} for mean in MEANS}
    # compared as printed, to one decimal
    all_rates = {mean: {key: pct(r["mean"] if key == RANDOM_LABEL else r["top1"]) for key, r in rates.items()}
                 for mean, rates in all_rates.items()}
    if all(len(set(rates.values())) == 1 for rates in all_rates.values()):
        all_units = same_or_both({mean: next(iter(rates.values())) for mean, rates in all_rates.items()}, str)
    else:
        all_units = "; ".join(f"{key} " + " / ".join(all_rates[mean][key] for mean in MEANS) for key, _ in keys) \
            + f" ({' / '.join(MEANS)})"
    print(f"- All units (k = all): {all_units}")
    print(f"- N test comments: {same_or_both({m: results[m]['data']['num_test_comments'] for m in MEANS}, str)}")
    ceiling = same_or_both({m: results[m]["ceiling"] for m in MEANS},
                           lambda c: f"{pct(c['top1'])} ({c['correct']}/{c['total']}, {c['source']})")
    print(f"- Ceiling: {ceiling}")
    print("- Sources: " + ", ".join(f"{mean} {results[mean]['file']}" for mean in MEANS))
    print()


def main():
    parser = argparse.ArgumentParser(description="zero-filled vs available-case matching results as markdown tables")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="subtoken")
    parser.add_argument("--n", choices=[str(n) for n in NGRAM_SIZES] + [COMBINED], default="1")
    args = parser.parse_args()
    n = args.n if args.n == COMBINED else int(args.n)
    results = {mean: load_latest(args.unit, n, mean) for mean in MEANS}
    print(f"## {args.unit} units, n = {n}: top-1 on test (%)\n")
    for method, label in METHODS:
        print_method(method, label, results)


if __name__ == "__main__":
    main()
