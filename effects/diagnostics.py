# effects, diagnostics for the matching results: why do the occlusion signatures (score 1,
# score 2) do worse than frequency? Nothing in the method changes; the signatures, unit counts,
# fitting and evaluation are effects.matching's own functions. Runs locally on CPU.
# Run from the repo root: python -m effects.diagnostics   (--unit: unit type, default subtoken;
# --n: n-gram size, default 1; for n >= 2 every "unit" below is an n-gram of units; --n 1-3: the
# combined matching result)
#
# Reads the latest matching_*.json and the scoring, deltas and cache files it names, and for
# k in KS (not "all"):
#   1. coverage: share of test comments that contain at least one of their own author's top-k units
#   2. overlap: per author, the units shared by each pair of signature types
#   3. per author: top-1 accuracy of each author's test comments at k = PER_AUTHOR_K, refitted as in
#      matching; each refit's overall top-1 must equal the matching JSON's
#   4. distinct units: size of the union of all authors' top-k units (the SVM features, from the
#      matching JSON)
#   5. n = 1-3 only: how many of the chosen units are n = 1, 2 and 3, averaged over authors
# Writes results/effects/{dataset}/{perturbation}/{model_mode}/{unit}/n{n}/diagnostics_{timestamp}.json
# and outputs/{dataset}/{perturbation}/{model_mode}/effects_diagnostics_{unit}_{timestamp}.html with
# one section per n that has diagnostics.

import os
import json
import html
import argparse
from itertools import combinations
from datetime import datetime
import torch
from runs import load_runs
from reporting.chart import PAGE_HEAD
from .units import UNIT_TYPES, NGRAM_SIZES, COMBINED, unit_key
from .unit_scores import (
    build_header_html, build_footer_html, section_heading, footer_prefix, save_page,
    unit_dir, latest_sections,
)
from .matching import (
    MATCHING_PREFIX, METHODS, top_signatures, random_signatures, load_val_average, load_train,
    load_test, fit, evaluate, signature_sources, SCORES, RANDOM_LABEL, signature_table_rows, roles_line,
    page_notes, load_combined_scorings, combined_signature_sources, merged_random_signatures,
    combined_unit_counts,
)

DIAGNOSTICS_PREFIX = "diagnostics_"

KS = [1, 2, 5, 10, 20, 50]
PER_AUTHOR_K = 10
# signature types and labels as in matching; random comes on top
SIGNATURES = SCORES
# every pair goes into the JSON, the page shows PAGE_PAIRS
PAIRS = list(combinations([key for key, _ in SIGNATURES], 2))
PAGE_PAIRS = [
    ("score1", "score2"), ("score1", "frequency"), ("score2", "frequency"),
]


# ---------- inputs ----------

def load_inputs(directory):
    # the latest matching result, and the scoring (scores, ranks) and corrected deltas it used
    runs = load_runs(directory, MATCHING_PREFIX)
    if not runs:
        raise SystemExit(f"no {MATCHING_PREFIX}*.json files in {directory} - run effects.matching first")
    matching = runs[-1]
    with open(os.path.join(directory, matching["scoring_file"])) as f:
        scoring = json.load(f)
    scoring["file"] = matching["scoring_file"]
    deltas = torch.load(os.path.join(directory, matching["deltas_file"]))
    return matching, scoring, deltas["corrected"].double()


# ---------- checks ----------
# sources: {signature type: ranked unit entries per author}, the same as matching.fit_all's;
# draw_random: (k, seed) -> random signatures, as in matching

def coverage(signatures, authors, test_counts, test_labels):
    # share of test comments containing at least one of their own author's signature units
    units = [{unit_key(entry["unit"]) for entry in signatures[author]} for author in authors]
    covered = sum(
        1 for (counts, _), label in zip(test_counts, test_labels.tolist())
        if any(unit in counts for unit in units[label])
    )
    return covered / len(test_counts)


def coverage_all(sources, draw_random, authors, test_counts, test_labels, seeds):
    result = {key: {} for key, _ in SIGNATURES}
    for k in KS:
        for key, _ in SIGNATURES:
            result[key][str(k)] = coverage(top_signatures(sources[key], authors, key, k), authors, test_counts, test_labels)
    result["random"] = {}
    for k in KS:
        per_seed = {
            str(seed): coverage(draw_random(k, seed), authors, test_counts, test_labels)
            for seed in seeds
        }
        result["random"][str(k)] = {"per_seed": per_seed, "mean": sum(per_seed.values()) / len(per_seed)}
    return result


def overlap_all(sources, authors):
    # {pair: {k: {"per_author": {author: shared units}, "mean": over authors}}}
    result = {}
    for first, second in PAIRS:
        pair = f"{first}-{second}"
        result[pair] = {}
        for k in KS:
            a = top_signatures(sources[first], authors, first, k)
            b = top_signatures(sources[second], authors, second, k)
            per_author = {
                author: len({unit_key(e["unit"]) for e in a[author]} & {unit_key(e["unit"]) for e in b[author]})
                for author in authors
            }
            result[pair][str(k)] = {"per_author": per_author, "mean": sum(per_author.values()) / len(authors)}
    return result


def ngram_mix_all(sources, authors):
    # n = 1-3 only: {signature type: {k: {n: chosen units of that n, averaged over authors}}}
    return {
        key: {
            str(k): {
                str(n): sum(1 for author in authors for e in signatures[author] if e["ngram"] == n) / len(authors)
                for n in NGRAM_SIZES
            }
            for k in KS for signatures in [top_signatures(sources[key], authors, key, k)]
        }
        for key, _ in SIGNATURES
    }


def per_author_all(sources, matching, authors, train, corrected_deltas, average, test):
    # top-1 per author at PER_AUTHOR_K: refit as in matching, evaluate each author's test comments
    # separately (still choosing among all authors); the overall top-1 must match the matching JSON
    embeddings, counts, labels = test
    result = {}
    for method, _ in METHODS:
        result[method] = {}
        for key, _ in SIGNATURES:
            signatures = top_signatures(sources[key], authors, key, PER_AUTHOR_K)
            fitted = fit(method, signatures, authors, train, corrected_deltas)
            overall = evaluate(method, fitted, test, average)
            saved = matching["results"][method][key][str(PER_AUTHOR_K)]
            if (overall["correct"], overall["total"]) != (saved["correct"], saved["total"]):
                raise SystemExit(f"refit {method}, {key}, k = {PER_AUTHOR_K}: top-1 {overall['correct']}/{overall['total']} "
                                 f"differs from the matching JSON's {saved['correct']}/{saved['total']}")
            per_author = {}
            for i, author in enumerate(authors):
                rows = (labels == i).nonzero().squeeze(1)
                subset = (embeddings[rows], [counts[r] for r in rows.tolist()], labels[rows])
                per_author[author] = evaluate(method, fitted, subset, average)
            result[method][key] = {"overall": overall, "per_author": per_author}
    return result


def distinct_units_all(matching):
    # number of distinct units in the union of all authors' top-k units = the SVM's features
    svm_results = matching["results"]["tf_svm"]
    result = {key: {str(k): svm_results[key][str(k)]["num_features"] for k in KS} for key, _ in SIGNATURES}
    result["random"] = {}
    for k in KS:
        per_seed = svm_results["random"][str(k)]["per_seed"]
        result["random"][str(k)] = sum(r["num_features"] for r in per_seed.values()) / len(per_seed)
    return result


# ---------- page ----------

def build_table_html(header, rows):
    # header: column titles (first one for the row label); rows: lists of cells, first one the label
    numeric = 'padding: 2px 8px; text-align: right; font-variant-numeric: tabular-nums;'
    head_cell = 'padding: 2px 8px; font-weight: 600; border-bottom: 1px solid #ccc;'
    head = f'<th style="{head_cell} text-align: left;">{header[0]}</th>' + "".join(
        f'<th style="{head_cell} text-align: right;">{title}</th>' for title in header[1:]
    )
    body = "".join(
        f'<tr><td style="padding: 2px 8px;">{row[0]}</td>'
        + "".join(f'<td style="{numeric}">{cell}</td>' for cell in row[1:])
        + "</tr>"
        for row in rows
    )
    return f"""
<table style="border-collapse: collapse; font-size: 0.85em; margin-bottom: 40px;">
    <thead><tr>{head}</tr></thead>
    <tbody>{body}</tbody>
</table>"""


def build_section_html(n, sections):
    result = sections[n]
    sentence = 'style="max-width: 900px; line-height: 1.5;"'
    k_header = [f"k = {k}" for k in KS]
    labels = dict(SIGNATURES) | {"random": RANDOM_LABEL}

    coverage_rows = [
        [labels[key]] + [f"{result['coverage'][key][str(k)]:.1%}" for k in KS] for key, _ in SIGNATURES
    ] + [[labels["random"]] + [f"{result['coverage']['random'][str(k)]['mean']:.1%}" for k in KS]]

    overlap_rows = [
        [f"{labels[first]} – {labels[second]}"] + [f"{result['overlap'][f'{first}-{second}'][str(k)]['mean']:.1f}" for k in KS]
        for first, second in PAGE_PAIRS
    ]

    distinct_rows = [
        [labels[key]] + [str(result["distinct_units"][key][str(k)]) for k in KS] for key, _ in SIGNATURES
    ] + [[labels["random"]] + [f"{result['distinct_units']['random'][str(k)]:.1f}" for k in KS]]

    per_author = result["per_author"]
    author_header = ["author"] + [label for _, label in SIGNATURES]
    author_tables = {
        method: [
            [html.escape(author)] + [f"{per_author[method][key]['per_author'][author]['top1']:.1%}" for key, _ in SIGNATURES]
            for author in result["authors"]
        ] + [["all"] + [f"<b>{per_author[method][key]['overall']['top1']:.1%}</b>" for key, _ in SIGNATURES]]
        for method, _ in METHODS
    }
    per_author_k = result['settings']['per_author_k']
    per_author_html = "".join(f"""
<h3>Per author, {method_label} (k = {per_author_k})</h3>
<p {sentence}>{method_label}: top-1 accuracy on each author's own test comments (the prediction still chooses among all
authors), refitted as in matching; "all" is the overall top-1, equal to the matching JSON.</p>
{build_table_html(author_header, author_tables[method])}""" for method, method_label in METHODS)

    return f"""
{section_heading(n)}

<h3>Coverage</h3>
<p {sentence}>Share of the {result['num_test_comments']} test comments that contain at least one of their own author's top-k units.</p>
{build_table_html(["signature"] + k_header, coverage_rows)}

<h3>Overlap</h3>
<p {sentence}>Number of units two signature types share in an author's top k, averaged over the {len(result['authors'])} authors.</p>
{build_table_html(["pair"] + k_header, overlap_rows)}

<h3>Distinct units across authors</h3>
<p {sentence}>Number of different units in the union of all authors' top-k units (= the TF-SVM's features; at most
{len(result['authors'])} × k, reached only if no two authors share a unit), from the matching JSON.</p>
{build_table_html(["signature"] + k_header, distinct_rows)}
{build_mix_html(result, sentence, k_header, labels)}
{per_author_html}"""


def build_mix_html(result, sentence, k_header, labels):
    # n = 1-3 only: how many of the chosen units are n = 1, 2 and 3
    if "ngram_mix" not in result:
        return ""
    rows = [
        [labels[key]] + [" / ".join(f"{result['ngram_mix'][key][str(k)][str(n)]:.1f}" for n in NGRAM_SIZES) for k in KS]
        for key, _ in SIGNATURES
    ]
    return f"""
<h3>Units per n</h3>
<p {sentence}>Of an author's top-k units, how many are n = 1 / n = 2 / n = 3, averaged over the {len(result['authors'])} authors.</p>
{build_table_html(["signature"] + k_header, rows)}"""


def page_sections(unit):
    # {n: latest diagnostics result} of the unit type; files written before the data counts were
    # stored get them from the matching file they name (in memory only)
    sections = latest_sections(unit, DIAGNOSTICS_PREFIX)
    for n, result in sections.items():
        if "data" not in result:
            with open(os.path.join(unit_dir(unit, n), result["matching_file"])) as f:
                result["data"] = json.load(f)["data"]
    return sections


def section_inputs(result):
    # the input files of one section, for the footer
    if result.get("ngram") == COMBINED:
        per_n = "; ".join(
            f"n = {n}: {html.escape(files['scoring_file'])}, {html.escape(files['deltas_file'])}, "
            f"{', '.join(html.escape(name) for name in files['cache_val_files'] + files['cache_test_files'])} "
            f"(cache series {html.escape(files['cache_series'])})"
            for n, files in result["inputs"].items()
        )
        return (f"input {html.escape(result['file'])}, {html.escape(result['matching_file'])}; {per_n}; "
                f"test embeddings and val average from n = 1")
    return "input " + ", ".join(
        html.escape(name) for name in [result["file"], result["matching_file"], result["scoring_file"], result["deltas_file"]]
        + result["cache_val_files"] + result["cache_test_files"]
    ) + f"; cache series {html.escape(result['cache_series'])}"


def build_page(unit, sections):
    # the page of one unit type; sections: {n: diagnostics result}, one section per n-gram size, each
    # with its file name under "file"
    first = next(iter(sections.values()))
    footer = [footer_prefix(n) + section_inputs(result) for n, result in sections.items()]
    return f"""<html>
<head>{PAGE_HEAD}</head>
<body>
{build_header_html(f"matching diagnostics - {html.escape(first['dataset'])}, {html.escape(unit)} units", first, roles_line(first["data"]),
                   signature_table_rows(len(first["settings"]["random_seeds"])), page_notes(sections))}
{"".join(build_section_html(n, sections) for n in sections)}
{build_footer_html(footer)}
</body>
</html>"""


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="diagnostics for the effects matching results")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="subtoken")
    parser.add_argument("--n", choices=[str(n) for n in NGRAM_SIZES] + [COMBINED], default="1")
    return parser.parse_args()


def single_inputs(unit, n):
    # everything the checks need for one n, from the latest matching result and the files it names
    directory = unit_dir(unit, n)
    matching, scoring, corrected_deltas = load_inputs(directory)
    authors = matching["authors"]
    print(f"matching {matching['file']}: scoring {matching['scoring_file']}, deltas {matching['deltas_file']}, "
          f"cache series {matching['cache_series']}")
    average, _ = load_val_average(directory, scoring)
    train = load_train(authors, unit, n)
    test_embeddings, test_counts, test_labels, test_files = load_test(directory, matching["cache_series"], authors, unit, n)
    if test_files != matching["cache_test_files"]:
        raise SystemExit(f"cache test files {test_files} differ from the matching JSON's {matching['cache_test_files']}")
    return {
        "matching": matching,
        "sources": signature_sources(scoring, matching["frequency_ranking"]),
        "draw_random": lambda k, seed: random_signatures(scoring["scores"], authors, k, seed),
        "corrected_deltas": corrected_deltas,
        "average": average,
        "train": train,
        "test": (test_embeddings, test_counts, test_labels),
        "files": {key: matching[key] for key in ("scoring_file", "deltas_file", "cache_series", "cache_val_files", "cache_test_files")},
    }


def combined_inputs(unit):
    # the same for n = 1-3, from the latest combined matching result and the single-n files it names
    directory = unit_dir(unit, COMBINED)
    runs = load_runs(directory, MATCHING_PREFIX)
    if not runs:
        raise SystemExit(f"no {MATCHING_PREFIX}*.json files in {directory} - run effects.matching --n {COMBINED} first")
    matching = runs[-1]
    print(f"matching {matching['file']} (n = {COMBINED})")
    scorings, corrected_deltas, offsets = load_combined_scorings(
        unit, {n: matching["inputs"][str(n)]["scoring_file"] for n in NGRAM_SIZES})
    authors = matching["authors"]
    frequencies = {n: matching["frequency_rankings"][str(n)] for n in NGRAM_SIZES}
    sources, scores_by_n = combined_signature_sources(scorings, frequencies, authors, offsets)
    average, _ = load_val_average(unit_dir(unit, 1), scorings[1])
    count = lambda text: combined_unit_counts(text, unit)
    train = load_train(authors, unit, count=count)
    test_embeddings, test_counts, test_labels, test_files = load_test(
        unit_dir(unit, 1), scorings[1]["cache_series"], authors, unit, count=count)
    if test_files != matching["inputs"]["1"]["cache_test_files"]:
        raise SystemExit(f"cache test files {test_files} differ from the matching JSON's {matching['inputs']['1']['cache_test_files']}")
    return {
        "matching": matching,
        "sources": sources,
        "draw_random": lambda k, seed: merged_random_signatures(scores_by_n, authors, k, seed),
        "corrected_deltas": corrected_deltas,
        "average": average,
        "train": train,
        "test": (test_embeddings, test_counts, test_labels),
        "files": {"inputs": matching["inputs"]},
    }


def main():
    args = parse_args()
    unit = args.unit
    n = args.n if args.n == COMBINED else int(args.n)
    created_at = datetime.now()
    stamp = f"{created_at:%Y%m%d-%H%M%S}"
    directory = unit_dir(unit, n)

    inputs = combined_inputs(unit) if n == COMBINED else single_inputs(unit, n)
    matching, sources, test = inputs["matching"], inputs["sources"], inputs["test"]
    authors = matching["authors"]
    seeds = matching["settings"]["random_seeds"]
    _, test_counts, test_labels = test

    result = {
        "experiment": "effects_diagnostics",
        "created_at": created_at.isoformat(timespec="seconds"),
        "matching_file": matching["file"],
        **inputs["files"],
        "dataset": matching["dataset"],
        "model": matching["model"],
        "mode": matching["mode"],
        "perturbation": matching["perturbation"],
        "unit": unit,
        "ngram": n,
        "authors": authors,
        "num_test_comments": len(test_labels),
        # val, train and test counts of the matching run
        "data": matching["data"],
        "settings": {
            "ks": KS,
            "per_author_k": PER_AUTHOR_K,
            "random_seeds": seeds,
            "matching_settings": matching["settings"],
        },
        # {signature: {k: share}}, random: {k: {"per_seed", "mean"}}
        "coverage": coverage_all(sources, inputs["draw_random"], authors, test_counts, test_labels, seeds),
        # {"first-second": {k: {"per_author", "mean"}}}
        "overlap": overlap_all(sources, authors),
        # {signature: {k: distinct units}}, random: mean over seeds; from the matching JSON's num_features
        "distinct_units": distinct_units_all(matching),
        # {method: {signature: {"overall", "per_author": {author: top-1}}}} at per_author_k
        "per_author": per_author_all(sources, matching, authors, inputs["train"], inputs["corrected_deltas"],
                                     inputs["average"], test),
    }
    if n == COMBINED:
        # {signature: {k: {n: chosen units of that n, averaged over authors}}}
        result["ngram_mix"] = ngram_mix_all(sources, authors)

    result_path = os.path.join(directory, f"{DIAGNOSTICS_PREFIX}{stamp}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"saved results to {result_path}")

    # the unit type's page, with every n that has diagnostics, this one included
    save_page(f"effects_diagnostics_{unit}", build_page(unit, page_sections(unit)), stamp)


if __name__ == "__main__":
    main()
