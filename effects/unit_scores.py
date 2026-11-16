# effects, step 2: two scores per author and cached unit, from the saved effects.cache embeddings
# only, and every author's units ranked by them. Runs locally on CPU: no model is loaded and
# nothing is embedded.
# Run from the repo root: python -m effects.unit_scores   (--unit: unit type, default subtoken;
# --n: n-gram size, default 1; --mean: background mean, zero-filled (default) or available-case)
#
# Reads the latest effects.cache val series (it must be complete) in
# results/effects/{dataset}/{perturbation}/{model_mode}/{unit}/n{n}/ for config.py's dataset,
# perturbation, model and mode, and writes scoring_{timestamp}.json (all scores and ranks) and
# deltas_{timestamp}.pt (raw and corrected effect vectors) to the mean's folder (result_dir: the
# cache folder for zero-filled, its available-case/ subfolder for available-case), plus the page
# outputs/{dataset}/{perturbation}/{model_mode}/effects_scoring_{unit}_{mean}_{timestamp}.html with
# one section per n that has scores of that mean. A "unit" below is an n-gram of units for n >= 2.
#
# Per author A and unit a cached for A, over C(A,a) = A's sampled comments that contain a (n of them):
#   score 1 = 1 - cos(mean emb(text), mean emb(text without a))
#   delta   = mean (emb(text) - emb(text without a)), score 2 = |delta|
# Backgrounds per unit, over authors B incl. A: bg1 = mean_B score1(B,a), bg_delta = mean_B delta(B,a);
#   zero-filled: over all authors, a not cached for B counts as 0 / zero vector;
#   available-case: over only the authors B for whom a is cached
#   score 1 corrected = score 1 - bg1, score 2 corrected = |delta - bg_delta|
# final = corrected * n / (n + d), d = DAMPING, units ranked by final score, highest first.
# score1_d20: score 1 with d = DAMPING_D20 instead.

import os
import json
import html
import argparse
from datetime import datetime
import torch
import config
from reporting.chart import PAGE_HEAD
from runs import results_dir, load_runs, latest_series
from .units import UNIT_TYPES, NGRAM_SIZES, SECTION_NS, display_unit, unit_key

# as in effects.cache, which isn't imported here since it loads LUAR
RESULTS_ROOT = "results/effects"
VAL_PREFIX = "val_"

SCORING_PREFIX = "scoring_"
DELTAS_PREFIX = "deltas_"
OUTPUTS_ROOT = "outputs"
# chosen constants d: final = corrected * n / (n + d), damps units found in few comments
DAMPING = 5
DAMPING_D20 = 20
TABLE_ROWS = 20

# (score key, column title); each ranks by f"{key}_final"
SCORES = [
    ("score1", "score 1"),
    ("score1_d20", "score 1 (d = 20)"),
    ("score2", "score 2"),
]
# rows of the signature table at the top of the effects pages: key -> (label, ranked by, d)
SIGNATURE_ROWS = {
    "score1": ("score 1", "1 − cos(mean emb, mean emb without unit), minus mean over authors", DAMPING),
    "score1_d20": ("score 1 (d = 20)", "same", DAMPING_D20),
    "score2": ("score 2", "|Δ − mean Δ over authors|", DAMPING),
}
DAMPING_NOTE = "damping: × n / (n + d), n = author's comments containing the unit"
# the background mean (--mean) of bg1, bg_delta and matching's bgf: zero-filled = over all authors, 0
# where the unit isn't cached for an author; available-case = over only the authors where it is cached
MEANS = ["zero-filled", "available-case"]
DEFAULT_MEAN = "zero-filled"
MEAN_NOTES = {
    "zero-filled": "background mean (zero-filled): over all authors, 0 where the unit isn't cached for an author",
    "available-case": "background mean (available-case): over only the authors where the unit is cached",
}
BACKGROUND_DEFINITIONS = {
    "zero-filled": {
        "bg1": "mean over all authors B of score1_raw(B,a), 0 where a is not cached for B",
        "bg_delta": "mean over all authors B of delta(B,a), the zero vector where a is not cached for B",
    },
    "available-case": {
        "bg1": "mean of score1_raw(B,a) over only the authors B for whom a is cached",
        "bg_delta": "mean of delta(B,a) over only the authors B for whom a is cached",
    },
}
# the raw score shown next to each final score; score1_d20 has no raw score of its own
RAW_KEYS = {"score1": "score1_raw", "score1_d20": "score1_raw", "score2": "score2_raw"}

def load_cache(directory):
    # the latest effects.cache val series: (series id, its runs oldest first, and per sampled
    # comment (author, doc_index) -> (its metadata, the embeddings of the run file it is in))
    series, runs = latest_series(load_runs(directory, VAL_PREFIX))
    if series is None:
        raise SystemExit(f"no {VAL_PREFIX}*.json files in {directory} - run effects.cache first")
    comments = {}
    for run in runs:
        embeddings = torch.load(os.path.join(directory, run["embeddings_file"]))
        for doc in run["documents"]:
            comments[(doc["author"], doc["doc_index"])] = (doc, embeddings)
    num_planned = len(runs[-1]["plan"])
    if len(comments) < num_planned:
        raise SystemExit(f"cache series {series} is incomplete: {len(comments)}/{num_planned} val comments done - "
                         f"run effects.cache again")
    return series, runs, comments


def gather_units(comments):
    # {author: {unit: {"author_count", "base": [emb(text)], "removed": [emb(text without unit)]}}},
    # one base/removed pair per comment of C(author, unit)
    per_author = {}
    for (author, _), (doc, embeddings) in comments.items():
        units = per_author.setdefault(author, {})
        for entry in doc["units"]:
            unit = units.setdefault(unit_key(entry["unit"]), {"author_count": entry["author_count"], "base": [], "removed": []})
            unit["base"].append(embeddings[doc["base_row"]])
            unit["removed"].append(embeddings[entry["row"]])
    return per_author


def raw_scores(gathered):
    # score 1, delta and score 2 of one author's unit, in float64 from the float32 cache
    base = torch.stack(gathered["base"]).double()
    removed = torch.stack(gathered["removed"]).double()
    score1 = 1 - torch.nn.functional.cosine_similarity(base.mean(dim=0), removed.mean(dim=0), dim=0).item()
    delta = (base - removed).mean(dim=0)
    return score1, delta


def compute_scores(per_author, authors, ngram=1, mean=DEFAULT_MEAN):
    # {author: [unit entries]} with all scores and ranks, and the raw and corrected deltas;
    # each author's entries are ordered by their score 1 rank; n-gram entries (ngram >= 2) carry
    # their n-gram size under "ngram" ("n" is the number of comments); mean: the background mean
    raw = {
        author: {unit: raw_scores(gathered) for unit, gathered in per_author.get(author, {}).items()}
        for author in authors
    }

    # backgrounds: sums over the authors the unit is cached for, divided by all authors (zero-filled:
    # a unit not cached for an author counts as 0 / zero vector) or by those authors (available-case)
    bg1 = {}
    bg_delta = {}
    cached_for = {}
    for author in authors:
        for unit, (score1, delta) in raw[author].items():
            bg1[unit] = bg1.get(unit, 0.0) + score1
            bg_delta[unit] = bg_delta.get(unit, torch.zeros_like(delta)) + delta
            cached_for[unit] = cached_for.get(unit, 0) + 1
    divisor = {unit: len(authors) if mean == "zero-filled" else count for unit, count in cached_for.items()}
    bg1 = {unit: total / divisor[unit] for unit, total in bg1.items()}
    bg_delta = {unit: total / divisor[unit] for unit, total in bg_delta.items()}

    scores = {}
    raw_deltas = []
    corrected_deltas = []
    for author in authors:
        entries = []
        for unit, (score1, delta) in raw[author].items():
            n = len(per_author[author][unit]["base"])
            damping = n / (n + DAMPING)
            corrected_delta = delta - bg_delta[unit]
            entry = {
                "unit": unit,
                **({"ngram": ngram} if ngram > 1 else {}),
                "n": n,
                "author_count": per_author[author][unit]["author_count"],
                "score1_raw": score1,
                "bg1": bg1[unit],
                "score1_corrected": score1 - bg1[unit],
                "score1_final": (score1 - bg1[unit]) * damping,
                "score2_raw": delta.norm().item(),
                "score2_corrected": corrected_delta.norm().item(),
                "score2_final": corrected_delta.norm().item() * damping,
                "score1_d20_corrected": score1 - bg1[unit],
                "score1_d20_final": (score1 - bg1[unit]) * n / (n + DAMPING_D20),
            }
            entries.append((entry, delta, corrected_delta))

        # highest final score first, ties by unit string
        for key, _ in SCORES:
            order = sorted(entries, key=lambda e: (-e[0][f"{key}_final"], e[0]["unit"]))
            for rank, (entry, _, _) in enumerate(order, start=1):
                entry[f"rank_{key}"] = rank

        scores[author] = []
        for entry, delta, corrected_delta in sorted(entries, key=lambda e: e[0]["rank_score1"]):
            entry["delta_row"] = len(raw_deltas)
            raw_deltas.append(delta)
            corrected_deltas.append(corrected_delta)
            scores[author].append(entry)

    deltas = {
        "raw": torch.stack(raw_deltas).to(torch.float32),
        "corrected": torch.stack(corrected_deltas).to(torch.float32),
    }
    return scores, deltas


def build_result(series, runs, scores, unit, created_at, deltas_file, ngram=1, mean=DEFAULT_MEAN):
    last = runs[-1]
    num_comments = {author: 0 for author in last["authors"]}
    for author, _ in last["plan"]:
        num_comments[author] += 1
    return {
        "experiment": "effects_unit_scores",
        "created_at": created_at.isoformat(timespec="seconds"),
        "cache_series": series,
        "cache_files": [run["file"] for run in runs],
        "cache_embeddings_files": [run["embeddings_file"] for run in runs],
        "dataset": last["dataset"],
        "model": last["model"],
        "mode": last["mode"],
        "perturbation": last["perturbation"],
        "unit": unit,
        "ngram": ngram,
        "mean": mean,
        "cache_settings": last["settings"],
        "settings": {
            "damping": DAMPING,
            "damping_d20": DAMPING_D20,
            "num_authors": len(last["authors"]),
            "rank_order": "final score, highest first; ties by unit string",
            "computed_in": "float64 from the float32 cache; deltas saved as float32",
        },
        "definitions": {
            "C(A,a)": "author A's sampled comments that contain unit a; n = |C(A,a)|",
            "score1_raw": "1 - cos(mean over C(A,a) of emb(text), mean over C(A,a) of emb(text without a))",
            "delta": "mean over C(A,a) of (emb(text) - emb(text without a))",
            "score2_raw": "length of delta(A,a)",
            **BACKGROUND_DEFINITIONS[mean],
            "score1_corrected": "score1_raw - bg1",
            "score2_corrected": "length of delta(A,a) - bg_delta(a)",
            "final": f"corrected * n / (n + d), d = {DAMPING}",
            "score1_d20_corrected": "score1_corrected (the same value)",
            "score1_d20_final": f"score1_d20_corrected * n / (n + {DAMPING_D20}); differs from score1 only in the damping",
        },
        "authors": last["authors"],
        "num_comments": num_comments,
        "deltas_file": deltas_file,
        "deltas_tensors": {"raw": "delta(A,a)", "corrected": "delta(A,a) - bg_delta(a)", "rows": "delta_row of each unit entry"},
        # {author: [unit entries, ordered by rank_score1]}
        "scores": scores,
    }


def format_number(value):
    # real minus sign (U+2212) for negative numbers
    s = f"{value:.4f}"
    return "−" + s[1:] if s.startswith("-") else s


def build_table_html(entries, key, unit):
    numeric = 'padding: 2px 8px; text-align: right; font-variant-numeric: tabular-nums;'
    columns = ["final", "raw", "n", "count"]
    head = (
        f'<th style="{numeric} font-weight: 600; border-bottom: 1px solid #ccc;">#</th>'
        f'<th style="padding: 2px 8px; text-align: left; font-weight: 600; border-bottom: 1px solid #ccc;">unit</th>'
        + "".join(
            f'<th style="{numeric} font-weight: 600; border-bottom: 1px solid #ccc;">{column}</th>'
            for column in columns
        )
    )
    ranked = sorted(entries, key=lambda e: e[f"rank_{key}"])
    body = ""
    for entry in ranked[:TABLE_ROWS]:
        values = [format_number(entry[f"{key}_final"]), format_number(entry[RAW_KEYS[key]]), str(entry["n"]), str(entry["author_count"])]
        body += (
            f'<tr><td style="{numeric}">{entry[f"rank_{key}"]}</td>'
            f'<td style="padding: 2px 8px; font-family: ui-monospace, Menlo, monospace; white-space: pre;">{html.escape(display_unit(entry["unit"], unit))}</td>'
            # the first column is the one the units are ranked by
            + "".join(
                f'<td style="{numeric}{" font-weight: 600;" if i == 0 else ""}">{value}</td>'
                for i, value in enumerate(values)
            )
            + "</tr>"
        )
    if not entries:
        body = f'<tr><td colspan="{len(columns) + 2}" style="padding: 2px 8px; color: #666;">no cached units</td></tr>'
    return f"""
        <table style="border-collapse: collapse; font-size: 0.85em;">
            <thead><tr>{head}</tr></thead>
            <tbody>{body}</tbody>
        </table>"""


def build_author_row_html(author, entries, num_comments, unit):
    cells = ""
    for key, title in SCORES:
        cells += f"""
    <div>
        <div style="font-weight: 600; margin-bottom: 4px;">{title}</div>
        {build_table_html(entries, key, unit)}
    </div>"""
    return f"""
<h3 style="margin-bottom: 8px;">{html.escape(author)} <span style="color:#666; font-weight:400; font-size: 0.8em;">({num_comments} comments, {len(entries)} cached units)</span></h3>
<div style="display: flex; gap: 40px; align-items: flex-start; margin-bottom: 32px;">{cells}
</div>"""


def build_header_html(title, result, roles, rows, notes):
    # the plain description block of the effects pages: title, setting, data roles, a table of the
    # signature types (rows: (label, ranked by, d)), one-line notes
    num_authors = len(result["authors"])
    cell = "padding: 4px 24px 4px 0; text-align: left;"
    head = f"{cell} border-bottom: 1px solid #ccc;"
    # a font list, not the bare generic monospace, so browsers don't shrink it below the text size
    formula = "font-family: ui-monospace, Menlo, Consolas, monospace; font-size: inherit;"
    table_rows = "".join(
        f'<tr><td style="{cell}">{label}</td><td style="{cell} {formula}">{html.escape(ranked_by)}</td>'
        f'<td style="{cell} text-align: right;">{d}</td></tr>'
        for label, ranked_by, d in rows
    )
    return f"""
<h2>{title}</h2>
<p>{html.escape(result['dataset'])}, {num_authors} authors, unit: {html.escape(result['unit'])}, model: {html.escape(result['model'])},
mode: {html.escape(result['mode'])}, perturbation: {html.escape(result['perturbation'])}</p>
<p>{roles}</p>
<table style="border-collapse: collapse;">
<tr><th style="{head}">signature</th><th style="{head}">ranked by</th><th style="{head} text-align: right;">d</th></tr>
{table_rows}
</table>
<p>{"<br>".join(notes)}</p>"""


def build_footer_html(lines):
    return f'<p style="color: #888; font-size: 0.8em;">{"<br>".join(lines)}</p>'


def section_heading(n):
    # a page has one section per n-gram size of its unit type
    return f"<h2>n = {n}</h2>"


def footer_prefix(n):
    return f"n = {n}: "


def unit_dir(unit, n):
    # the cache folder of the unit type and n; the cache is always read from here
    return os.path.join(results_dir(RESULTS_ROOT, config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME), unit, f"n{n}")


def result_dir(unit, n, mean=DEFAULT_MEAN):
    # where the results of a background mean are written and looked up: the cache folder for
    # zero-filled, its subfolder named after the mean otherwise
    if mean not in MEANS:
        raise ValueError(f"unknown mean {mean!r}, known: {MEANS}")
    directory = unit_dir(unit, n)
    return directory if mean == "zero-filled" else os.path.join(directory, mean)


def latest_sections(unit, prefix, mean=DEFAULT_MEAN):
    # {n: the latest result file of prefix for the unit type, n and mean (in the mean's folder only)},
    # in order n = 1, 2, 3, 1-3, for every n that has one (result files are written complete); each
    # with its file name under "file"
    sections = {}
    for n in SECTION_NS:
        runs = load_runs(result_dir(unit, n, mean), prefix)
        if runs:
            sections[n] = runs[-1]
    return sections


def save_page(name, page, stamp):
    output_dir = os.path.join(OUTPUTS_ROOT, config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_{stamp}.html")
    with open(output_path, "w") as f:
        f.write(page)
    print(f"saved page to {output_path}")
    return output_path


def build_page(unit, sections, mean=DEFAULT_MEAN):
    # the page of one unit type and mean; sections: {n: scoring result}, one section per n-gram size,
    # each with its file name under "file"
    first = next(iter(sections.values()))
    body = "".join(
        section_heading(n) + "".join(
            build_author_row_html(author, result["scores"][author], result["num_comments"][author], unit)
            for author in result["authors"]
        )
        for n, result in sections.items()
    )
    footer = [
        footer_prefix(n) + f"input {', '.join(html.escape(name) for name in [result['file']] + result['cache_files'] + result['cache_embeddings_files'])}; "
        f"cache series {html.escape(result['cache_series'])}"
        for n, result in sections.items()
    ]
    return f"""<html>
<head>{PAGE_HEAD}</head>
<body>
{build_header_html(f"unit scores - {html.escape(first['dataset'])}, {html.escape(unit)} units", first,
                   f"val {sum(first['num_comments'].values())} (signatures)",
                   [SIGNATURE_ROWS[key] for key, _ in SCORES], [DAMPING_NOTE, MEAN_NOTES[mean]])}
{body}
{build_footer_html(footer)}
</body>
</html>"""


def parse_args():
    parser = argparse.ArgumentParser(description="unit scores per author from the effects cache")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="subtoken")
    parser.add_argument("--n", type=int, choices=NGRAM_SIZES, default=1)
    parser.add_argument("--mean", choices=MEANS, default=DEFAULT_MEAN)
    return parser.parse_args()


def main():
    args = parse_args()
    unit, n, mean = args.unit, args.n, args.mean
    created_at = datetime.now()
    stamp = f"{created_at:%Y%m%d-%H%M%S}"
    cache_dir = unit_dir(unit, n)
    directory = result_dir(unit, n, mean)
    os.makedirs(directory, exist_ok=True)
    series, runs, comments = load_cache(cache_dir)
    print(f"cache series {series}: {len(comments)} comments from {', '.join(run['file'] for run in runs)}")

    authors = runs[-1]["authors"]
    scores, deltas = compute_scores(gather_units(comments), authors, n, mean)

    deltas_file = f"{DELTAS_PREFIX}{stamp}.pt"
    deltas_path = os.path.join(directory, deltas_file)
    torch.save(deltas, deltas_path)
    print(f"saved {deltas['raw'].shape[0]} raw and corrected deltas to {deltas_path}")

    result = build_result(series, runs, scores, unit, created_at, deltas_file, n, mean)
    result_path = os.path.join(directory, f"{SCORING_PREFIX}{stamp}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"saved scores to {result_path}")

    # the unit type's page of this mean, with every n that has scores of it, this one included
    save_page(f"effects_scoring_{unit}_{mean}", build_page(unit, latest_sections(unit, SCORING_PREFIX, mean), mean), stamp)


if __name__ == "__main__":
    main()
