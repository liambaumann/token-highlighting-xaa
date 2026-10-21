# author token signatures from the saved signature.scores files only - no model needed.
# Run from the repo root: python -m signature.tokens
# Reads the latest scores series in the results folder of config.py's dataset, perturbation,
# model and mode, and for every author and every token string in that author's scored comments
# computes sum (of its occlusion scores over all its positions), comments (that contain it),
# positions (occurrences) and mean (sum / positions). Each author's tokens are then ranked three
# ways (see METHODS): by sum, by mean over tokens with at least MIN_POSITIONS positions, and by
# frequency (comments).
# Writes results/.../tokens_{timestamp}.json (all tokens of every author and the full rankings)
# and outputs/{dataset}/{perturbation}/{model_mode}/signature_tokens_{timestamp}.html.

import os
import json
import html
from datetime import datetime
import config
from reporting.chart import PAGE_HEAD
from .results import results_dir, latest_series, series_documents

SCORES_PREFIX = "scores_"
TOKENS_PREFIX = "tokens_"
OUTPUTS_ROOT = "outputs"
TABLE_ROWS = 20
# chosen setting: the mean ranking only considers tokens with at least this many positions
MIN_POSITIONS = 10

# (key in the results JSON, column title, columns shown per row, the first one is ranked by)
METHODS = [
    ("sum", "sum", ["sum", "comments", "positions"]),
    (f"mean_min{MIN_POSITIONS}", f"mean, at least {MIN_POSITIONS} positions", ["mean", "comments", "positions"]),
    ("frequency", "frequency", ["comments", "positions"]),
]

METHOD_DEFINITIONS = {
    "sum": "sum of the token's occlusion scores over all its positions in the author's comments, highest first; ties by token string",
    f"mean_min{MIN_POSITIONS}": f"sum divided by positions, highest first, only tokens with at least {MIN_POSITIONS} positions "
                                f"({MIN_POSITIONS} is a chosen setting); ties by token string",
    "frequency": "number of the author's comments that contain the token, highest first; ties by positions (highest first), "
                 "then token string; no occlusion scores involved",
}

# how a removed token is filled in, per config.PERTURBATION (see model.embed_variants)
PERTURBATION_DESCRIPTIONS = {
    "doc_mean": "replaced by doc_mean",
    "mask": "replaced by the tokenizer's mask token",
    "delete": "deleted from the comment",
}


def load_score_runs(directory):
    # every signature.scores run file in the folder, oldest first (file names are start
    # timestamps); each run gets its file name under "file"
    if not os.path.isdir(directory):
        return []
    runs = []
    for fname in sorted(os.listdir(directory)):
        if fname.startswith(SCORES_PREFIX) and fname.endswith(".json"):
            with open(os.path.join(directory, fname)) as f:
                run = json.load(f)
            run["file"] = fname
            runs.append(run)
    return runs


def author_signatures(documents, authors):
    # {author: [{"token", "sum", "comments", "positions", "mean"}, ...]} over all token strings of
    # the author's comments, highest sum first (ties by token string); <s> and </s> are never
    # occluded, so only content positions count
    stats = {author: {} for author in authors}
    for doc in documents:
        author_stats = stats[doc["author"]]
        for position in range(1, len(doc["tokens"]) - 1):
            token = doc["tokens"][position]
            entry = author_stats.setdefault(token, {"sum": 0.0, "comments": set(), "positions": 0})
            entry["sum"] += doc["scores"][position]
            entry["comments"].add(doc["doc_index"])
            entry["positions"] += 1

    signatures = {}
    for author in authors:
        rows = [
            {
                "token": token,
                "sum": entry["sum"],
                "comments": len(entry["comments"]),
                "positions": entry["positions"],
                "mean": entry["sum"] / entry["positions"],
            }
            for token, entry in stats[author].items()
        ]
        signatures[author] = sorted(rows, key=lambda row: (-row["sum"], row["token"]))
    return signatures


def rank(rows, method):
    # the rows of one author ordered by a METHODS key, as defined in METHOD_DEFINITIONS
    if method == "sum":
        return sorted(rows, key=lambda row: (-row["sum"], row["token"]))
    if method == f"mean_min{MIN_POSITIONS}":
        eligible = [row for row in rows if row["positions"] >= MIN_POSITIONS]
        return sorted(eligible, key=lambda row: (-row["mean"], row["token"]))
    if method == "frequency":
        return sorted(rows, key=lambda row: (-row["comments"], -row["positions"], row["token"]))
    raise ValueError(f"unknown ranking method {method!r}")


def build_result(series, series_runs, created_at):
    documents = series_documents(series_runs)
    last = series_runs[-1]
    authors = last["authors"]
    comments_per_author = {author: 0 for author in authors}
    for doc in documents:
        comments_per_author[doc["author"]] += 1
    signatures = author_signatures(documents, authors)
    return {
        "experiment": "signature_tokens",
        "created_at": created_at.isoformat(timespec="seconds"),
        "scores_series": series,
        "scores_files": [run["file"] for run in series_runs],
        "scores_complete": len(documents) >= len(last["plan"]),
        "dataset": last["dataset"],
        "model": last["model"],
        "mode": last["mode"],
        "perturbation": last["perturbation"],
        "scores_settings": last["settings"],
        "authors": authors,
        "num_comments": comments_per_author,
        "min_positions": MIN_POSITIONS,
        "methods": METHOD_DEFINITIONS,
        # every token of every author with its four numbers, highest sum first
        "signatures": signatures,
        # {author: {method: [token, ...]}}, each method's full ranking as token strings
        "rankings": {
            author: {method: [row["token"] for row in rank(signatures[author], method)] for method, _, _ in METHODS}
            for author in authors
        },
    }


def format_value(column, value):
    return f"{value:.4f}" if column in ("sum", "mean") else str(value)


def build_table_html(rows, columns):
    numeric = 'padding: 2px 8px; text-align: right; font-variant-numeric: tabular-nums;'
    head = (
        f'<th style="{numeric} font-weight: 600; border-bottom: 1px solid #ccc;">#</th>'
        f'<th style="padding: 2px 8px; text-align: left; font-weight: 600; border-bottom: 1px solid #ccc;">token</th>'
        + "".join(
            f'<th style="{numeric} font-weight: 600; border-bottom: 1px solid #ccc;">{column}</th>'
            for column in columns
        )
    )
    body = ""
    for position, row in enumerate(rows[:TABLE_ROWS], start=1):
        body += (
            f'<tr><td style="{numeric}">{position}</td>'
            f'<td style="padding: 2px 8px; font-family: ui-monospace, Menlo, monospace; white-space: pre;">{html.escape(row["token"])}</td>'
            # the first column is the one the method ranks by
            + "".join(
                f'<td style="{numeric}{" font-weight: 600;" if i == 0 else ""}">{format_value(column, row[column])}</td>'
                for i, column in enumerate(columns)
            )
            + "</tr>"
        )
    if not rows:
        body = f'<tr><td colspan="{len(columns) + 2}" style="padding: 2px 8px; color: #666;">no tokens</td></tr>'
    return f"""
        <table style="border-collapse: collapse; font-size: 0.85em;">
            <thead><tr>{head}</tr></thead>
            <tbody>{body}</tbody>
        </table>"""


def build_author_row_html(author, signature, num_comments):
    cells = ""
    for method, title, columns in METHODS:
        ranked = rank(signature, method)
        cells += f"""
    <div>
        <div style="font-weight: 600; margin-bottom: 4px;">{title} <span style="color:#666; font-weight:400;">({len(ranked)} tokens ranked)</span></div>
        {build_table_html(ranked, columns)}
    </div>"""
    return f"""
<h3 style="margin-bottom: 8px;">{html.escape(author)} <span style="color:#666; font-weight:400; font-size: 0.8em;">({num_comments} comments, {len(signature)} distinct tokens)</span></h3>
<div style="display: flex; gap: 40px; align-items: flex-start; margin-bottom: 32px;">{cells}
</div>"""


def build_description_html(result):
    settings = result["scores_settings"]
    authors = result["authors"]
    counts = sorted(set(result["num_comments"].values()))
    per_author = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
    status = "" if result["scores_complete"] else (
        f" Incomplete: {sum(result['num_comments'].values())} of "
        f"{len(authors) * settings['comments_per_author']} planned comments."
    )
    files = ", ".join(html.escape(name) for name in result["scores_files"])
    perturbation = PERTURBATION_DESCRIPTIONS.get(result["perturbation"], f"perturbed by {html.escape(result['perturbation'])}")
    return f"""
<div style="max-width: 900px; line-height: 1.5; margin-bottom: 32px;">
<p>{html.escape(result['dataset'])} {html.escape(settings['split'])}, {len(authors)} authors × {per_author} comments
(seed {settings['sample_seed']}); {html.escape(result['model'])}, {html.escape(result['mode'])}, {html.escape(result['perturbation'])},
n = {settings['n']}.{status}<br>
<b>score</b> = cos(comment, author centroid) − cos(comment with token {perturbation}, author centroid)<br>
<b>sum</b> = Σ scores over the token's positions · <b>mean</b> = sum / positions, only tokens with ≥ {MIN_POSITIONS} positions
(chosen setting) · <b>frequency</b> = comments (ties: positions), no scores<br>
<b>comments</b> = author's comments containing the token · <b>positions</b> = total occurrences ·
bold = ranked by, top {TABLE_ROWS}, highest first<br>
LUAR tokens: Ġ = leading space, Ċ = line break, ĊĊ = empty line · source: {files}</p>
</div>"""


def build_page(result):
    rows_html = "".join(
        build_author_row_html(author, result["signatures"][author], result["num_comments"][author])
        for author in result["authors"]
    )
    return f"""<html>
<head>{PAGE_HEAD}</head>
<body>
<h1>author token signatures - {html.escape(result['dataset'])}</h1>
{build_description_html(result)}
{rows_html}
</body>
</html>"""


def main():
    created_at = datetime.now()
    scores_dir = results_dir(config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME)
    series, series_runs = latest_series(load_score_runs(scores_dir))
    if series is None:
        print(f"no {SCORES_PREFIX}*.json files in {scores_dir} - run signature.scores first")
        return

    result = build_result(series, series_runs, created_at)
    if not result["scores_complete"]:
        print(f"warning: scores series {series} is incomplete, signatures use the comments done so far")

    result_path = os.path.join(scores_dir, f"{TOKENS_PREFIX}{created_at:%Y%m%d-%H%M%S}.json")
    with open(result_path, "w") as f:
        json.dump(result, f)
    print(f"saved signatures to {result_path}")

    output_dir = os.path.join(OUTPUTS_ROOT, config.DATASET, config.PERTURBATION, config.MODEL_MODE_NAME)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"signature_tokens_{created_at:%Y%m%d-%H%M%S}.html")
    with open(output_path, "w") as f:
        f.write(build_page(result))
    print(f"saved page to {output_path}")


if __name__ == "__main__":
    main()
