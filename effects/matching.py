# effects, step 3: can the step-2 signatures (each author's top k units by score 1 or score 2)
# attribute unseen comments? Two matchers, against a random-units baseline and the retrieval
# ceiling. Runs locally on CPU; uses saved embeddings only, nothing is embedded.
# Run from the repo root: python -m effects.matching   (--unit: unit type, default subtoken;
# --n: n-gram size, default 1; for n >= 2 every "unit" below is an n-gram of units)
#
# Data roles: val (the cached comments) gives the signatures, the Δ-vectors and the average
# embedding; train (all train comments) only trains the SVMs; test (all test comments) is only
# read in the evaluation, after everything is fitted.
#
# TF-SVM: features = the union of all authors' top-k units; a comment's value for a unit is its
#   occurrences divided by the comment's number of units. One LinearSVC (one-vs-rest, balanced
#   classes) trained on train; a test comment goes to the author with the highest margin.
# Δ-sum: author profile = sum of the corrected Δ (Δ - background) of its top-k units, normalised
#   to length 1; a test comment's embedding minus the val average embedding goes to the profile
#   with the highest cosine similarity.
# Random baseline: per author k units drawn at random from its cached units, RANDOM_SEEDS seeds.
# Frequency: reference baseline, a ranking of the same cached units by damped, background-corrected
#   relative frequency in the author's val comments (frequency_ranking).
# Damping: final = corrected * n / (n + d), n = the author's comments containing the unit.
# Ceiling: top-1 accuracy of the latest saved retrieval run (config.RETRIEVAL_RESULTS_DIR).
#
# Writes results/effects/{dataset}/{perturbation}/{model_mode}/{unit}/n{n}/matching_{timestamp}.json
# and outputs/{dataset}/{perturbation}/{model_mode}/effects_matching_{unit}_{timestamp}.html with one
# section per n that has matching results.

import os
import json
import html
import random
import argparse
import warnings
from collections import Counter
from datetime import datetime
import torch
import scipy.sparse
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import config
from data import DATASETS, load_jsonl_by_author, load_test_texts
from metrics import top1_accuracy
from runs import load_runs
from reporting.chart import PAGE_HEAD, OVERVIEW_LEGEND_PLUGIN_OPTIONS, TOP_RGB, BOTTOM_RGB, RANDOM_RGB, build_note_html
from .units import UNIT_TYPES, NGRAM_SIZES, comment_units, unit_key
from .unit_scores import (
    DAMPING, DAMPING_D20, SIGNATURE_ROWS, DAMPING_NOTE, build_header_html, build_footer_html, section_heading,
    footer_prefix, save_page, unit_dir, latest_sections,
)
from .cache import TEST_PREFIX, prepare_comment

SCORING_PREFIX = "scoring_"
MATCHING_PREFIX = "matching_"

KS = [1, 2, 5, 10, 20, 50, "all"]
# (signature type, label); score types rank by the scoring file's rank_{key}, frequency types by their own ranking
SCORES = [
    ("score1", "score 1"),
    ("score1_d20", "score 1 (d = 20)"),
    ("score2", "score 2"),
    ("frequency", "frequency"),
]
RANDOM_LABEL = "random"
CEILING_LABEL = "full LUAR (ceiling)"
RANDOM_SEEDS = list(range(50))
SVM_RANDOM_STATE = 0

METHODS = [
    ("tf_svm", "TF-SVM"),
    ("delta_sum", "Δ-sum"),
]
FREQUENCY_RGB = (138, 79, 200)
# (RGB, dashed) per signature type; d = 20 dashed in the colour of its d = 5 type
SCORE_STYLES = {
    "score1": (TOP_RGB, False),
    "score1_d20": (TOP_RGB, True),
    "score2": (BOTTOM_RGB, False),
    "frequency": (FREQUENCY_RGB, False),
}
CEILING_RGB = (60, 60, 60)


# ---------- inputs ----------

def load_ceiling():
    # top-1 of the latest saved retrieval run for config.py's dataset, model and mode
    directory = config.RETRIEVAL_RESULTS_DIR
    files = sorted(f for f in os.listdir(directory) if f.startswith("test_") and f.endswith(".json")) if os.path.isdir(directory) else []
    if not files:
        raise SystemExit(f"no retrieval result (test_*.json) in {directory} - run retrieval first (./vsc.sh retrieval)")
    path = os.path.join(directory, files[-1])
    with open(path) as f:
        run = json.load(f)
    if (run["dataset"], run["model"], run["mode"]) != (config.DATASET, config.MODEL, config.MODE):
        raise SystemExit(f"{path} is for {run['dataset']}, {run['model']}, {run['mode']}, not config.py's "
                         f"{config.DATASET}, {config.MODEL}, {config.MODE}")
    top1 = run["metrics"]["top1"]
    return {"top1": top1["rate"], "correct": top1["correct"], "total": top1["total"], "source": path,
            "dataset": run["dataset"], "model": run["model"], "mode": run["mode"], "datetime": run["datetime"]}


def load_scoring(directory):
    # the latest unit_scores result and its corrected deltas
    runs = load_runs(directory, SCORING_PREFIX)
    if not runs:
        raise SystemExit(f"no {SCORING_PREFIX}*.json files in {directory} - run effects.unit_scores first")
    scoring = runs[-1]
    deltas = torch.load(os.path.join(directory, scoring["deltas_file"]))
    return scoring, deltas["corrected"].double()


def load_val_average(directory, scoring):
    # mean of the base embeddings of the val comments in the cache files the scoring used
    base = {}
    for fname in scoring["cache_files"]:
        with open(os.path.join(directory, fname)) as f:
            run = json.load(f)
        embeddings = torch.load(os.path.join(directory, run["embeddings_file"]))
        for doc in run["documents"]:
            base[(doc["author"], doc["doc_index"])] = embeddings[doc["base_row"]]
    return torch.stack(list(base.values())).double().mean(dim=0), len(base)


def unit_counts(text, unit, ngram=1):
    # occurrences of every unit (n-gram) in the comment, and its number of units (n-grams);
    # tokenized and split exactly as effects.cache does (first 510 content tokens)
    occurrences = prepare_comment(text, unit, ngram)["occurrences"]
    return Counter(occurrences), len(occurrences)


def frequency_ranking(directory, scoring, authors, unit, ngram=1):
    # reference baseline: per author its cached units (the ones the unit scores rank), ranked by
    # freq = occurrences in the author's sampled val comments / all units in them, minus
    # bgf = mean over all authors of freq (0 where never used), damped by n / (n + DAMPING),
    # n = the author's comments containing the unit. Units from the cached comments' tokens, as the
    # cache; for ngram >= 2 the units are n-grams and freq is over all n-grams.
    comments = {}
    for fname in scoring["cache_files"]:
        with open(os.path.join(directory, fname)) as f:
            run = json.load(f)
        for doc in run["documents"]:
            comments[(doc["author"], doc["doc_index"])] = doc["tokens"]

    occurrences = {author: Counter() for author in authors}
    comments_with = {author: Counter() for author in authors}
    num_units = {author: 0 for author in authors}
    for (author, _), tokens in comments.items():
        units = [name for name, _ in comment_units(tokens, unit, ngram)]
        occurrences[author].update(units)
        comments_with[author].update(set(units))
        num_units[author] += len(units)
    freq = {author: {name: count / num_units[author] for name, count in occurrences[author].items()} for author in authors}

    ranking = {}
    for author in authors:
        entries = []
        for cached in scoring["scores"][author]:
            name = unit_key(cached["unit"])
            bgf = sum(freq[b].get(name, 0.0) for b in authors) / len(authors)
            n = comments_with[author][name]
            corrected = freq[author][name] - bgf
            entries.append({
                "unit": name, "n": n, "freq": freq[author][name], "bgf": bgf, "corrected": corrected,
                "final": corrected * n / (n + DAMPING), "delta_row": cached["delta_row"],
            })
        # highest final first, ties by unit string
        entries.sort(key=lambda entry: (-entry["final"], entry["unit"]))
        for rank, entry in enumerate(entries, start=1):
            entry["rank_frequency"] = rank
        ranking[author] = entries
    return ranking


def signature_sources(scoring, frequency):
    # the ranked unit entries each signature type selects from
    return {"frequency": frequency} | {key: scoring["scores"] for key, _ in SCORES if key != "frequency"}


def load_train(authors, unit, ngram=1):
    # all train comments: (unit counts per comment, author index per comment)
    train = load_jsonl_by_author(DATASETS[config.DATASET]["train"])
    counts = []
    labels = []
    for i, author in enumerate(authors):
        for text in train[author]:
            counts.append(unit_counts(text, unit, ngram))
            labels.append(i)
    return counts, labels


def load_test(directory, series, authors, unit, ngram=1):
    # all test comments, in the order of the cache's test plan: their cached embeddings,
    # unit counts and author indices
    runs = [run for run in load_runs(directory, TEST_PREFIX) if run["series"] == series]
    if not runs:
        raise SystemExit(f"no {TEST_PREFIX}*.json files of cache series {series} in {directory}")
    rows = {}
    for run in runs:
        embeddings = torch.load(os.path.join(directory, run["embeddings_file"]))
        for doc in run["documents"]:
            rows[(doc["author"], doc["doc_index"])] = embeddings[doc["row"]]
    plan = runs[-1]["plan"]
    if len(rows) < len(plan):
        raise SystemExit(f"cache series {series}: only {len(rows)}/{len(plan)} test comments embedded - run effects.cache again")

    texts = {author: load_test_texts(config.DATASET, author) for author in authors}
    embeddings = torch.stack([rows[(author, doc_index)] for author, doc_index in plan]).double()
    counts = [unit_counts(texts[author][doc_index], unit, ngram) for author, doc_index in plan]
    labels = torch.tensor([authors.index(author) for author, _ in plan])
    return embeddings, counts, labels, [run["file"] for run in runs]


# ---------- signatures ----------

def top_signatures(scores, authors, key, k):
    # {author: unit entries}: each author's top k units by key's final score ("all": all of them)
    signatures = {}
    for author in authors:
        ranked = sorted(scores[author], key=lambda entry: entry[f"rank_{key}"])
        signatures[author] = ranked if k == "all" else ranked[:k]
    return signatures


def random_signatures(scores, authors, k, seed):
    # {author: unit entries}: k of each author's cached units drawn at random (one rng per seed and k)
    rng = random.Random(seed)
    signatures = {}
    for author in authors:
        entries = sorted(scores[author], key=lambda entry: entry["unit"])
        signatures[author] = entries if k == "all" else rng.sample(entries, min(k, len(entries)))
    return signatures


# ---------- matchers ----------

def tf_matrix(comment_counts, features):
    # rows: comments, columns: features; value = occurrences of the unit / the comment's units
    column = {unit: j for j, unit in enumerate(features)}
    rows, cols, values = [], [], []
    for i, (counts, num_units) in enumerate(comment_counts):
        for unit, count in counts.items():
            j = column.get(unit)
            if j is not None:
                rows.append(i)
                cols.append(j)
                values.append(count / num_units)
    return scipy.sparse.csr_matrix((values, (rows, cols)), shape=(len(comment_counts), len(features)))


def fit_tf_svm(signatures, train_counts, train_labels):
    features = sorted({unit_key(entry["unit"]) for entries in signatures.values() for entry in entries})
    svm = LinearSVC(class_weight="balanced", random_state=SVM_RANDOM_STATE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        svm.fit(tf_matrix(train_counts, features), train_labels)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {"features": features, "svm": svm, "converged": converged}


def fit_delta_sum(signatures, authors, corrected_deltas):
    # one profile per author (row i = authors[i]): sum of its units' corrected deltas, length 1
    profiles = torch.stack([
        corrected_deltas[[entry["delta_row"] for entry in signatures[author]]].sum(dim=0)
        for author in authors
    ])
    return {"profiles": profiles / profiles.norm(dim=1, keepdim=True)}


def evaluate(method, fitted, test, average):
    # top-1 accuracy on test: per comment one value per author (margin / cosine similarity),
    # the highest counts as the prediction
    embeddings, counts, labels = test
    if method == "tf_svm":
        svm = fitted["svm"]
        assert list(svm.classes_) == list(range(len(svm.classes_)))
        sims = torch.tensor(svm.decision_function(tf_matrix(counts, fitted["features"])))
    else:
        sims = torch.nn.functional.cosine_similarity(
            (embeddings - average).unsqueeze(1), fitted["profiles"].unsqueeze(0), dim=2
        )
    correct, total = top1_accuracy(sims, labels)
    return {"top1": correct / total, "correct": correct, "total": total}


def fit(method, signatures, authors, train, corrected_deltas):
    if method == "tf_svm":
        return fit_tf_svm(signatures, *train)
    return fit_delta_sum(signatures, authors, corrected_deltas)


def fit_info(method, fitted):
    return {"num_features": len(fitted["features"]), "converged": fitted["converged"]} if method == "tf_svm" else {}


# ---------- page ----------

def format_rate(value):
    return f"{value:.2%}"


def build_chart_html(chart_id, title, method_results, ceiling):
    labels = [str(k) for k in KS]
    datasets = []
    for key, label in SCORES:
        rgb, dashed = SCORE_STYLES[key]
        color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        datasets.append({
            "label": label,
            "data": [round(method_results[key][str(k)]["top1"], 4) for k in KS],
            "borderColor": color, "backgroundColor": color, "borderWidth": 2, "pointRadius": 3,
            **({"borderDash": [4, 3]} if dashed else {}),
        })
    random_results = method_results["random"]
    band_color = f"rgba({RANDOM_RGB[0]}, {RANDOM_RGB[1]}, {RANDOM_RGB[2]}, 0.2)"
    random_color = f"rgb({RANDOM_RGB[0]}, {RANDOM_RGB[1]}, {RANDOM_RGB[2]})"
    # min-max band of the random baseline (two unlabeled datasets, the second filled up to the first)
    datasets.append({
        "label": "", "data": [round(random_results[str(k)]["max"], 4) for k in KS],
        "borderColor": "transparent", "backgroundColor": band_color, "borderWidth": 0, "pointRadius": 0, "fill": False,
    })
    datasets.append({
        "label": "", "data": [round(random_results[str(k)]["min"], 4) for k in KS],
        "borderColor": "transparent", "backgroundColor": band_color, "borderWidth": 0, "pointRadius": 0, "fill": "-1",
    })
    datasets.append({
        "label": RANDOM_LABEL,
        "data": [round(random_results[str(k)]["mean"], 4) for k in KS],
        "borderColor": random_color, "backgroundColor": random_color, "borderWidth": 2, "pointRadius": 3,
    })
    ceiling_color = f"rgb({CEILING_RGB[0]}, {CEILING_RGB[1]}, {CEILING_RGB[2]})"
    datasets.append({
        "label": CEILING_LABEL,
        "data": [round(ceiling["top1"], 4) for _ in KS],
        "borderColor": ceiling_color, "backgroundColor": ceiling_color, "borderWidth": 2, "borderDash": [6, 4],
        "pointRadius": 0, "pointHoverRadius": 0,
    })

    return f"""
    <div style="width: 480px;">
        <h4>{title}</h4>
        <canvas id="{chart_id}" width="480" height="360" style="width: 480px; height: 360px;"></canvas>
    </div>
    <script>
    new Chart(document.getElementById("{chart_id}"), {{
        type: 'line',
        data: {{
            labels: {json.dumps(labels)},
            datasets: {json.dumps(datasets)}
        }},
        options: {{
            responsive: false,
            interaction: {{ mode: 'index', intersect: false }},
            plugins: {{
                legend: {OVERVIEW_LEGEND_PLUGIN_OPTIONS}
            }},
            scales: {{
                y: {{ min: 0, max: 1, title: {{ display: true, text: 'top-1 accuracy' }} }},
                x: {{ title: {{ display: true, text: 'k (units per author)' }} }}
            }}
        }}
    }});
    </script>
    """


def build_table_html(result):
    numeric = 'padding: 2px 8px; text-align: right; font-variant-numeric: tabular-nums;'
    head_cell = 'padding: 2px 8px; font-weight: 600; border-bottom: 1px solid #ccc;'
    head = (
        f'<th style="{head_cell} text-align: left;">method</th>'
        f'<th style="{head_cell} text-align: right;">k</th>'
        + "".join(f'<th style="{head_cell} text-align: right;">{label}</th>' for _, label in SCORES)
        + f'<th style="{head_cell} text-align: right;">{RANDOM_LABEL} mean</th>'
        f'<th style="{head_cell} text-align: right;">{RANDOM_LABEL} min</th>'
        f'<th style="{head_cell} text-align: right;">{RANDOM_LABEL} max</th>'
    )
    body = ""
    for method, method_label in METHODS:
        method_results = result["results"][method]
        for i, k in enumerate(KS):
            k = str(k)
            random_results = method_results["random"][k]
            border = ' style="border-top: 1px solid #ddd;"' if i == 0 else ""
            cells = [method_label if i == 0 else "", k]
            cells += [format_rate(method_results[key][k]["top1"]) for key, _ in SCORES]
            cells += [format_rate(random_results[name]) for name in ("mean", "min", "max")]
            body += f"<tr{border}>" + f'<td style="padding: 2px 8px;">{cells[0]}</td>' + "".join(
                f'<td style="{numeric}">{cell}</td>' for cell in cells[1:]
            ) + "</tr>"
    ceiling = result["ceiling"]
    body += (
        f'<tr style="border-top: 1px solid #ddd;"><td style="padding: 2px 8px;">{CEILING_LABEL}</td><td></td>'
        f'<td colspan="{len(SCORES) + 3}" style="{numeric} text-align: left;">{format_rate(ceiling["top1"])} '
        f'({ceiling["correct"]}/{ceiling["total"]})</td></tr>'
    )
    return f"""
<table style="border-collapse: collapse; font-size: 0.85em; margin-bottom: 40px;">
    <thead><tr>{head}</tr></thead>
    <tbody>{body}</tbody>
</table>"""


def signature_table_rows(num_seeds):
    # rows of the signature table on the matching and diagnostics pages
    return [SIGNATURE_ROWS[key] for key, _ in SCORES if key in SIGNATURE_ROWS] + [
        ("frequency", "share of unit in author's text, minus mean share over authors", DAMPING),
        (RANDOM_LABEL, f"k random cached units, {num_seeds} seeds", ""),
    ]


def roles_line(data):
    return (f"val {data['num_val_comments']} (signatures), train {data['num_train_comments']} (SVM), "
            f"test {data['num_test_comments']} (evaluation), top-1 on test")


METHOD_NOTES = [
    "TF-SVM: unit counts per comment, LinearSVC trained on train",
    "Δ-sum: sum of the units' corrected Δ, nearest to test emb minus mean val emb",
]


def build_section_html(unit, n, sections):
    result = sections[n]
    charts_html = "".join(
        build_chart_html(f"matching-{unit}-n{n}-{method}", label, result["results"][method], result["ceiling"])
        for method, label in METHODS
    )
    return f"""
{section_heading(n)}
{build_note_html("x: k units per author, y: top-1 accuracy on test")}
<div style="display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 40px;">
{charts_html}
</div>
{build_table_html(result)}"""


def build_page(unit, sections):
    # the page of one unit type; sections: {n: matching result}, one section per n-gram size, each
    # with its file name under "file"
    first = next(iter(sections.values()))
    footer = [
        footer_prefix(n) + f"input {html.escape(result['file'])}, {html.escape(result['scoring_file'])}, {html.escape(result['deltas_file'])}, "
        f"{', '.join(html.escape(name) for name in result['cache_val_files'] + result['cache_test_files'])}; "
        f"cache series {html.escape(result['cache_series'])}; ceiling: {html.escape(result['ceiling']['source'])}"
        for n, result in sections.items()
    ]
    return f"""<html>
<head>{PAGE_HEAD}</head>
<body>
{build_header_html(f"signature matching - {html.escape(first['dataset'])}, {html.escape(unit)} units", first, roles_line(first["data"]),
                   signature_table_rows(len(first["settings"]["random_seeds"])), METHOD_NOTES + [DAMPING_NOTE])}
{"".join(build_section_html(unit, n, sections) for n in sections)}
{build_footer_html(footer)}
</body>
</html>"""


# ---------- main ----------

def parse_args():
    parser = argparse.ArgumentParser(description="signature matching on test with the effects unit scores")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="subtoken")
    parser.add_argument("--n", type=int, choices=NGRAM_SIZES, default=1)
    return parser.parse_args()


def fit_all(sources, scoring, authors, train, corrected_deltas):
    # every fitted matcher, before anything of test is read:
    # {method: {score key or "random": {k: fitted, or for random {seed: fitted}}}}
    fitted = {method: {key: {} for key, _ in SCORES} | {"random": {}} for method, _ in METHODS}
    for k in KS:
        for key, _ in SCORES:
            signatures = top_signatures(sources[key], authors, key, k)
            for method, _ in METHODS:
                fitted[method][key][str(k)] = fit(method, signatures, authors, train, corrected_deltas)
        for seed in RANDOM_SEEDS:
            signatures = random_signatures(scoring["scores"], authors, k, seed)
            for method, _ in METHODS:
                fitted[method]["random"].setdefault(str(k), {})[str(seed)] = fit(method, signatures, authors, train, corrected_deltas)
        print(f"k = {k}: fitted")
    return fitted


def evaluate_all(fitted, test, average):
    results = {}
    for method, _ in METHODS:
        results[method] = {}
        for key, _ in SCORES:
            results[method][key] = {
                k: {**evaluate(method, model, test, average), **fit_info(method, model)}
                for k, model in fitted[method][key].items()
            }
        results[method]["random"] = {}
        for k, per_seed_models in fitted[method]["random"].items():
            per_seed = {
                seed: {**evaluate(method, model, test, average), **fit_info(method, model)}
                for seed, model in per_seed_models.items()
            }
            rates = [r["top1"] for r in per_seed.values()]
            results[method]["random"][k] = {
                "per_seed": per_seed, "mean": sum(rates) / len(rates), "min": min(rates), "max": max(rates),
            }
    return results


def main():
    args = parse_args()
    unit, n = args.unit, args.n
    created_at = datetime.now()
    stamp = f"{created_at:%Y%m%d-%H%M%S}"
    directory = unit_dir(unit, n)

    ceiling = load_ceiling()
    scoring, corrected_deltas = load_scoring(directory)
    authors = scoring["authors"]
    print(f"scoring {scoring['file']} (cache series {scoring['cache_series']}), ceiling {ceiling['top1']:.2%} from {ceiling['source']}")

    # fitting: val (signatures, deltas, average) and train only
    average, num_val = load_val_average(directory, scoring)
    frequency = frequency_ranking(directory, scoring, authors, unit, n)
    train = load_train(authors, unit, n)
    fitted = fit_all(signature_sources(scoring, frequency), scoring, authors, train, corrected_deltas)

    # evaluation: the only place test is read
    test_embeddings, test_counts, test_labels, test_files = load_test(directory, scoring["cache_series"], authors, unit, n)
    results = evaluate_all(fitted, (test_embeddings, test_counts, test_labels), average)

    result = {
        "experiment": "effects_matching",
        "created_at": created_at.isoformat(timespec="seconds"),
        "scoring_file": scoring["file"],
        "deltas_file": scoring["deltas_file"],
        "cache_series": scoring["cache_series"],
        "cache_val_files": scoring["cache_files"],
        "cache_test_files": test_files,
        "dataset": scoring["dataset"],
        "model": scoring["model"],
        "mode": scoring["mode"],
        "perturbation": scoring["perturbation"],
        "unit": unit,
        "ngram": n,
        "authors": authors,
        "settings": {
            "ks": KS,
            "signature_types": {key: label for key, label in SCORES},
            "damping": f"final = corrected * n / (n + d), n = the author's comments containing the unit; "
                       f"d = {DAMPING}, or {DAMPING_D20} for score1_d20",
            "frequency": "reference baseline, not from the unit scores: freq = occurrences of the unit in the author's "
                         "sampled val comments / all units in them; bgf = mean over all authors of freq (0 where never used); "
                         f"final = (freq - bgf) * n / (n + {DAMPING}); "
                         "candidates = the author's cached units; ties by unit string",
            "random_seeds": RANDOM_SEEDS,
            "random_draw": "per seed and k one random.Random(seed); per author in author order, k units sampled "
                           "from the author's cached units sorted by unit string (all of them at k = all)",
            "tf_features": "union of all authors' signature units; value = occurrences in the comment / the comment's number of units",
            "svm": {"model": "sklearn.svm.LinearSVC", "class_weight": "balanced", "multi_class": "ovr",
                    "random_state": SVM_RANDOM_STATE, "other_parameters": "defaults"},
            "delta_sum": "profile = normalised sum of corrected deltas of the signature units; "
                         "test embedding minus the val average embedding, highest cosine similarity",
            "scoring_settings": scoring["settings"],
            "cache_settings": scoring["cache_settings"],
        },
        "data": {"num_val_comments": num_val, "num_train_comments": len(train[1]), "num_test_comments": len(test_labels)},
        "ceiling": ceiling,
        # {author: [the author's cached units, ordered by rank_frequency]}
        "frequency_ranking": frequency,
        "results": results,
    }

    result_path = os.path.join(directory, f"{MATCHING_PREFIX}{stamp}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"saved results to {result_path}")

    # the unit type's page, with every n that has matching results, this one included
    save_page(f"effects_matching_{unit}", build_page(unit, latest_sections(unit, MATCHING_PREFIX)), stamp)

    for method, label in METHODS:
        for k in ("10", "all"):
            r = results[method]
            print(f"{label}, k = {k}: " + ", ".join(f"{name} {r[key][k]['top1']:.2%}" for key, name in SCORES) + ", "
                  f"random {r['random'][k]['mean']:.2%} [{r['random'][k]['min']:.2%}, {r['random'][k]['max']:.2%}]")


if __name__ == "__main__":
    main()
