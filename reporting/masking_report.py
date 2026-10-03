import os
import json
from model import tokenizer
import config
from .highlight import save_highlighted_html
from .chart import (
    build_overview_chart_html, no_results_chart_html, OVERVIEW_SELECTIONS,
    PAGE_HEAD, build_note_html,
)

RESULTS_ROOT = "results/masking"
OUTPUTS_ROOT = "outputs"

NS = [1, 2, 3]
OVERVIEW_ROW_MODES = ["plain", "chunk32"]

# matches run.py's budget_percentages = [0.01, 0.03, 0.05, 0.10], plus the 0%/no-removal point
X_LABELS = [0, 1, 3, 5, 10]


def compute_doc_drops(result):
    # base_similarity minus the masked similarity, so every document starts at 0 - same metric,
    # same x axis (% of tokens removed) used everywhere these charts appear
    base = result["base_similarity"]
    budgets = sorted(result["budget_results"], key=lambda b: b["pct"])
    return {
        key: [0.0] + [base - b[field] for b in budgets]
        for key, field in [("top", "top_similarity"), ("random_avg", "random_similarity_avg"), ("bottom", "bottom_similarity")]
    }


def compute_y_range(all_values):
    if not all_values:
        return (-0.1, 0.1)
    span = max(all_values) - min(all_values)
    pad = span * 0.05 if span > 0 else 0.05
    return (min(all_values) - pad, max(all_values) + pad)


def build_per_document_pages():
    # unchanged from before: one overview.html + highlight pages per {dataset}/{model_mode} folder
    dataset_names = []

    for dataset_name in sorted(os.listdir(RESULTS_ROOT)):
        dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        dataset_names.append(dataset_name)

        for model_mode_name in sorted(os.listdir(dataset_dir)):
            results_dir = os.path.join(dataset_dir, model_mode_name)
            if not os.path.isdir(results_dir):
                continue

            json_files = [fname for fname in os.listdir(results_dir) if fname.endswith(".json")]
            if not json_files:
                continue

            output_dir = os.path.join(OUTPUTS_ROOT, dataset_name, model_mode_name)
            os.makedirs(output_dir, exist_ok=True)

            results = []
            for fname in json_files:
                with open(os.path.join(results_dir, fname)) as f:
                    results.append(json.load(f))

            results.sort(key=lambda r: (r["author"], r["doc_index"], r["n"]))

            # same drop metric, x axis and y range for every chart on this page, so they're
            # directly comparable to each other (and to the aggregate masking_overview.html)
            drops_by_result = [compute_doc_drops(result) for result in results]
            page_values = [v for drops in drops_by_result for key, _, _ in OVERVIEW_SELECTIONS for v in drops[key]]
            y_range = compute_y_range(page_values)

            all_charts_html = ""

            for result, drops in zip(results, drops_by_result):
                author = result["author"]
                doc_index = result["doc_index"]
                n = result["n"]

                filename = f"{author}_{doc_index}"
                if n == 1:
                    highlight_path = save_highlighted_html(result["tokens"], result["scores"], filename, tokenizer, output_dir=output_dir)
                else:
                    highlight_path = os.path.join(output_dir, f"{filename}.html")

                chart_title = f"{author}_{doc_index}_n{n}"
                all_charts_html += build_overview_chart_html(
                    chart_id=f"chart-{chart_title}", title=chart_title, docs=[drops],
                    x_labels=X_LABELS, y_range=y_range, link_href=os.path.basename(highlight_path),
                )

            note_html = build_note_html("lines: top-k / random-k (avg) / bottom-k removed, as similarity drop from the base")

            overview_html = f"""<html>
<head>{PAGE_HEAD}</head>
<body>
{note_html}
{all_charts_html}
</body>
</html>"""

            overview_path = os.path.join(output_dir, "overview.html")
            with open(overview_path, "w") as f:
                f.write(overview_html)

            print(f"\nsaved overview to {overview_path}")

    return dataset_names


def load_mode_results(dataset_name, mode):
    model_mode_name = f"{config.MODEL}_{mode}"
    results_dir = os.path.join(RESULTS_ROOT, dataset_name, model_mode_name)
    if not os.path.isdir(results_dir):
        return {}

    by_n = {n: [] for n in NS}
    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            result = json.load(f)
        n = result["n"]
        if n not in by_n:
            continue

        by_n[n].append(compute_doc_drops(result))

    return by_n


def build_masking_overview(dataset_name):
    mode_data = {mode: load_mode_results(dataset_name, mode) for mode in OVERVIEW_ROW_MODES}

    all_values = [
        v
        for mode in OVERVIEW_ROW_MODES
        for n in NS
        for doc in mode_data[mode].get(n, [])
        for key, _, _ in OVERVIEW_SELECTIONS
        for v in doc[key]
    ]
    y_range = compute_y_range(all_values)

    note_html = build_note_html("lines: mean over documents, bands: middle 50% of documents")

    rows_html = ""
    for mode in OVERVIEW_ROW_MODES:
        model_mode_name = f"{config.MODEL}_{mode}"
        link = f"{model_mode_name}/overview.html"

        charts_html = ""
        for n in NS:
            docs = mode_data[mode].get(n, [])
            title = f"n = {n} ({len(docs)} documents)"
            if docs:
                charts_html += build_overview_chart_html(
                    chart_id=f"overview-{dataset_name}-{mode}-n{n}",
                    title=title, docs=docs, x_labels=X_LABELS, y_range=y_range,
                )
            else:
                charts_html += no_results_chart_html(title)

        rows_html += f"""
        <h2>{mode} - <a href="{link}">per-document pages</a></h2>
        {note_html}
        <div style="display: flex; gap: 24px; margin-bottom: 40px;">
            {charts_html}
        </div>
        """

    overview_html = f"""<html>
<head>{PAGE_HEAD}</head>
<body>
<h1>masking overview - {dataset_name}</h1>
{rows_html}
</body>
</html>"""

    output_dir = os.path.join(OUTPUTS_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    overview_path = os.path.join(output_dir, "masking_overview.html")
    with open(overview_path, "w") as f:
        f.write(overview_html)

    print(f"saved masking overview to {overview_path}")


if __name__ == "__main__":
    dataset_names = build_per_document_pages()
    for dataset_name in dataset_names:
        build_masking_overview(dataset_name)
