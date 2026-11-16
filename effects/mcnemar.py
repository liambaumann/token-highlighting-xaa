# effects: exact McNemar test between two matching results on the same test comments.
# Run from the repo root:
#   python -m effects.mcnemar --side1 FILE SIGNATURE METHOD K --side2 FILE SIGNATURE METHOD K
#   python -m effects.mcnemar --side1 FILE SIGNATURE METHOD --side2 FILE SIGNATURE METHOD --all-k
# FILE: a matching_{timestamp}.json written by effects.matching (with correct_by_comment);
# SIGNATURE: a signature type in the file (score1, score1_d20, score2, frequency);
# METHOD: tf_svm or delta_sum; K: one of the file's ks. --all-k: one row per k (K left out).
# The valid values are read from the file, so effects.matching (and the model) is not imported.
#
# Both files must come from the same test plan (same cache series), otherwise it stops.
# b = comments side 1 got right and side 2 wrong, c = the opposite;
# p = scipy.stats.binomtest(b, b + c).pvalue (two-sided; 1 when b + c = 0).
# Prints k | acc 1 | acc 2 | b | c | p. Writes nothing.

import json
import argparse
from scipy.stats import binomtest


def test_series(result):
    # the cache series of the test plan the result was evaluated on (n = 1-3: the n = 1 cache's)
    return result["cache_series"] if "cache_series" in result else result["inputs"]["1"]["cache_series"]


def parse_side(values, all_k, name):
    # [file, signature, method] + [k] unless --all-k
    expected = 3 if all_k else 4
    if len(values) != expected:
        raise SystemExit(f"--{name}: expected FILE SIGNATURE METHOD{'' if all_k else ' K'}, got {' '.join(values)}")
    path, signature, method = values[:3]
    with open(path) as f:
        result = json.load(f)
    methods = list(result["results"])
    signatures = list(result["settings"]["signature_types"])
    ks = [str(k) for k in result["settings"]["ks"]]
    if method not in methods:
        raise SystemExit(f"--{name}: method {method!r} not one of {', '.join(methods)}")
    if signature not in signatures:
        raise SystemExit(f"--{name}: signature {signature!r} not one of {', '.join(signatures)}")
    k = None if all_k else values[3]
    if k is not None and k not in ks:
        raise SystemExit(f"--{name}: k {k!r} not one of {', '.join(ks)}")
    return {"path": path, "signature": signature, "method": method, "k": k, "ks": ks, "result": result}


def entry(side, k):
    found = side["result"]["results"][side["method"]][side["signature"]][k]
    if "correct_by_comment" not in found:
        raise SystemExit(f"{side['path']} has no correct_by_comment - rerun effects.matching")
    return found


def compare(entry1, entry2):
    hits1, hits2 = entry1["correct_by_comment"], entry2["correct_by_comment"]
    if len(hits1) != len(hits2):
        raise SystemExit(f"different numbers of test comments: {len(hits1)} vs {len(hits2)}")
    b = sum(1 for x, y in zip(hits1, hits2) if x and not y)
    c = sum(1 for x, y in zip(hits1, hits2) if y and not x)
    p = binomtest(b, b + c).pvalue if b + c else 1.0
    return {"acc1": entry1["top1"], "acc2": entry2["top1"], "b": b, "c": c, "p": p}


def main():
    parser = argparse.ArgumentParser(description="exact McNemar test between two effects matching results")
    parser.add_argument("--side1", nargs="+", required=True, metavar="ARG", help="FILE SIGNATURE METHOD [K]")
    parser.add_argument("--side2", nargs="+", required=True, metavar="ARG", help="FILE SIGNATURE METHOD [K]")
    parser.add_argument("--all-k", action="store_true", help="one row per k; K is left out of both sides")
    args = parser.parse_args()
    side1 = parse_side(args.side1, args.all_k, "side1")
    side2 = parse_side(args.side2, args.all_k, "side2")

    series1, series2 = test_series(side1["result"]), test_series(side2["result"])
    if series1 != series2:
        raise SystemExit(f"different test plans: cache series {series1} ({side1['path']}) vs {series2} ({side2['path']})")

    for i, side in enumerate((side1, side2), start=1):
        r = side["result"]
        print(f"side {i}: {side['path']} - {side['signature']}, {side['method']}, "
              f"{r['unit']}, n = {r.get('ngram', '?')}, mean {r.get('mean', 'zero-filled')}")
    print(f"test plan: cache series {series1}")
    print()

    if args.all_k and side1["ks"] != side2["ks"]:
        raise SystemExit(f"different ks: {side1['ks']} vs {side2['ks']}")
    ks = side1["ks"] if args.all_k else [side1["k"]]
    rows = []
    for k in ks:
        k2 = k if args.all_k else side2["k"]
        label = k if k == k2 else f"{k} / {k2}"
        rows.append((label, compare(entry(side1, k), entry(side2, k2))))

    width = max(len("k"), *(len(label) for label, _ in rows))
    print(f"{'k':>{width}} | {'acc 1':>7} | {'acc 2':>7} | {'b':>5} | {'c':>5} | {'p':>9}")
    print(f"{'-' * width}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 5}-+-{'-' * 5}-+-{'-' * 9}")
    for label, r in rows:
        print(f"{label:>{width}} | {r['acc1']:>7.2%} | {r['acc2']:>7.2%} | {r['b']:>5} | {r['c']:>5} | {r['p']:>9.3g}")


if __name__ == "__main__":
    main()
