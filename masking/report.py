import os
import json
from model import tokenizer
import config
from .highlight import save_highlighted_html
from reporting.chart import (
    build_overview_chart_html, no_results_chart_html, OVERVIEW_SELECTIONS,
    PAGE_HEAD, build_note_html,
)

RESULTS_ROOT = "results/masking"
OUTPUTS_ROOT = "outputs"

NS = [1, 2, 3]
OVERVIEW_ROW_MODES = ["plain", "chunk32"]
PERTURBATIONS = ["delete", "mask", "doc_mean"]

# matches masking/run_masking.py's budget_percentages = [0.01, 0.03, 0.05, 0.10], plus the 0%/no-removal point
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
    # one overview.html + highlight pages per {dataset}/{perturbation}/{model_mode} folder
    dataset_names = []

    for dataset_name in sorted(os.listdir(RESULTS_ROOT)):
        dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        dataset_names.append(dataset_name)

        for perturbation_name in sorted(os.listdir(dataset_dir)):
            perturbation_dir = os.path.join(dataset_dir, perturbation_name)
            if not os.path.isdir(perturbation_dir):
                continue

            for model_mode_name in sorted(os.listdir(perturbation_dir)):
                results_dir = os.path.join(perturbation_dir, model_mode_name)
                if not os.path.isdir(results_dir):
                    continue

                json_files = [fname for fname in os.listdir(results_dir) if fname.endswith(".json")]
                if not json_files:
                    continue

                output_dir = os.path.join(OUTPUTS_ROOT, dataset_name, perturbation_name, model_mode_name)
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


def load_mode_results(dataset_name, perturbation, mode):
    model_mode_name = f"{config.MODEL}_{mode}"
    results_dir = os.path.join(RESULTS_ROOT, dataset_name, perturbation, model_mode_name)
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


SELECTION_LABELS = {"top": "top", "random_avg": "random", "bottom": "bottom"}


TABLE_STYLE = """
<style>
    .pt-table { border-collapse: collapse; margin-bottom: 24px; font-size: 0.9em; width: auto; }
    .pt-table th, .pt-table td { padding: 4px 12px; text-align: left; }
    .pt-table thead tr { border-bottom: 2px solid #999; }
    .pt-table .pt-num { text-align: right; font-variant-numeric: tabular-nums; }
    .pt-table td[rowspan] { vertical-align: top; }
    .pt-table tbody tr.pt-group-start { border-top: 1px solid #ddd; }
    .pt-table tr.pt-muted { color: #888; }
</style>
"""


def fmt_num(v):
    # real minus sign (U+2212) instead of a hyphen, for negative numbers
    s = f"{v:.4f}"
    return "−" + s[1:] if s.startswith("-") else s


def build_perturbation_table(mode_data):
    # one row per (mode, n, selection); mode/n/docs shown once per group of three (rowspan)
    rows_html = ""
    first_group = True
    for mode in OVERVIEW_ROW_MODES:
        for n in NS:
            docs = mode_data[mode].get(n, [])
            group_class = "" if first_group else " class=\"pt-group-start\""
            first_group = False

            if not docs:
                rows_html += f"""
                <tr{group_class}>
                    <td>{mode}</td><td>{n}</td><td colspan="8">no results</td>
                </tr>
                """
                continue

            num_docs = len(docs)
            for sel_idx, (key, _, _) in enumerate(OVERVIEW_SELECTIONS):
                # budget indices into the drops arrays: x_labels = [0, 1, 3, 5, 10] -> 1,3,5,10% at indices 1-4
                budget_means = [sum(doc[key][b] for doc in docs) / num_docs for b in range(1, 5)]
                aopc = sum(budget_means) / len(budget_means)

                if key == "top":
                    # top_drop > random_drop at the same budget/doc <=> top_similarity < random_similarity_avg
                    count = sum(1 for doc in docs if doc["top"][4] > doc["random_avg"][4])
                    comparison = f"{count}/{num_docs} ({count / num_docs:.0%})"
                else:
                    comparison = ""

                lead_cols = f'<td rowspan="3">{mode}</td><td rowspan="3">{n}</td><td rowspan="3">{num_docs}</td>' if sel_idx == 0 else ""
                budget_cells = "".join(f'<td class="pt-num">{fmt_num(v)}</td>' for v in budget_means)
                # only the top row of a group gets the group-separator border; random/bottom are muted
                row_class = group_class if sel_idx == 0 else ' class="pt-muted"'

                rows_html += f"""
                <tr{row_class}>
                    {lead_cols}
                    <td>{SELECTION_LABELS[key]}</td>
                    {budget_cells}
                    <td class="pt-num">{fmt_num(aopc)}</td>
                    <td class="pt-num">{comparison}</td>
                </tr>
                """

    return f"""
    <table class="pt-table">
        <thead>
            <tr>
                <th>mode</th>
                <th>n</th>
                <th>docs</th>
                <th>selection</th>
                <th class="pt-num">1%</th>
                <th class="pt-num">3%</th>
                <th class="pt-num">5%</th>
                <th class="pt-num">10%</th>
                <th class="pt-num">AOPC</th>
                <th class="pt-num">top &gt; random at 10%</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    """


def build_masking_overview(dataset_name):
    # one section per perturbation that has results, in a fixed order; a perturbation with
    # nothing under it gets no section at all
    perturbation_mode_data = {}
    for perturbation in PERTURBATIONS:
        mode_data = {mode: load_mode_results(dataset_name, perturbation, mode) for mode in OVERVIEW_ROW_MODES}
        has_results = any(mode_data[mode].get(n) for mode in OVERVIEW_ROW_MODES for n in NS)
        if has_results:
            perturbation_mode_data[perturbation] = mode_data

    # one shared y range across every chart on the whole page, not just within a section
    all_values = [
        v
        for mode_data in perturbation_mode_data.values()
        for mode in OVERVIEW_ROW_MODES
        for n in NS
        for doc in mode_data[mode].get(n, [])
        for key, _, _ in OVERVIEW_SELECTIONS
        for v in doc[key]
    ]
    y_range = compute_y_range(all_values)

    note_html = build_note_html("lines: mean over documents, bands: middle 50% of documents")

    sections_html = ""
    for perturbation, mode_data in perturbation_mode_data.items():
        table_html = build_perturbation_table(mode_data)
        rows_html = ""
        for mode in OVERVIEW_ROW_MODES:
            model_mode_name = f"{config.MODEL}_{mode}"
            link = f"{perturbation}/{model_mode_name}/overview.html"

            charts_html = ""
            for n in NS:
                docs = mode_data[mode].get(n, [])
                title = f"n = {n} ({len(docs)} documents)"
                if docs:
                    charts_html += build_overview_chart_html(
                        chart_id=f"overview-{dataset_name}-{perturbation}-{mode}-n{n}",
                        title=title, docs=docs, x_labels=X_LABELS, y_range=y_range,
                    )
                else:
                    charts_html += no_results_chart_html(title)

            rows_html += f"""
            <h3>{mode} - <a href="{link}">per-document pages</a></h3>
            {note_html}
            <div style="display: flex; gap: 24px; margin-bottom: 40px;">
                {charts_html}
            </div>
            """

        sections_html += f"""
        <h2>{perturbation}</h2>
        {table_html}
        {rows_html}
        """

    overview_html = f"""<html>
<head>{PAGE_HEAD}{TABLE_STYLE}</head>
<body>
<h1>masking overview - {dataset_name}</h1>
{sections_html}
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
