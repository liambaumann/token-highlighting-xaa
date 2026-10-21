# plots for the keep-top-k experiment, from saved results only - no model needed.
# Run from the repo root: python -m keep_top_k.plot
# One page per results/keep_top_k/{dataset}/{perturbation}/{model_mode} folder, written to
# outputs/{dataset}/{perturbation}/{model_mode}/keep_top_k.html, plotting that folder's latest series.

import os
import json
from reporting.chart import PAGE_HEAD, OVERVIEW_LEGEND_PLUGIN_OPTIONS, RANDOM_RGB, build_note_html
from runs import load_runs, latest_series, series_documents
from .keep import RESULTS_ROOT, SETTINGS, aggregate

OUTPUTS_ROOT = "outputs"

# (setting, RGB, legend label); blue and orange from reporting/chart.py, order checked for
# colour-vision-deficiency separation
SETTING_STYLES = [
    ("n1", (42, 120, 214), "n = 1"),
    ("n2", (235, 104, 52), "n = 2"),
    ("n3", (138, 79, 200), "n = 3"),
    ("n1-3", (31, 158, 106), "n = 1-3"),
]
BASELINE_RGB = RANDOM_RGB

# (metric key, chart title, y axis label, fixed y range or None for a range from the data)
METRICS = [
    ("similarity", "similarity to author centroid", "cosine similarity", None),
    ("top1", "retrieval top-1", "top-1 accuracy", (0, 1)),
    ("map", "retrieval MAP", "MAP", (0, 1)),
]


def compute_y_range(values):
    span = max(values) - min(values)
    pad = span * 0.05 if span > 0 else 0.05
    return (round(min(values) - pad, 4), round(max(values) + pad, 4))


def build_keep_chart_html(chart_id, title, y_label, summary, metric, y_range, max_k):
    ks = list(range(1, max_k + 1))
    datasets = []
    for setting, rgb, label in SETTING_STYLES:
        color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
        datasets.append({
            "label": label,
            "data": [{"x": row["k"], "y": round(row[metric], 4)} for row in summary["curves"][setting]],
            "borderColor": color,
            "backgroundColor": color,
            "borderWidth": 2,
            "pointRadius": 0,
            "pointHoverRadius": 4,
        })

    baseline_color = f"rgb({BASELINE_RGB[0]}, {BASELINE_RGB[1]}, {BASELINE_RGB[2]})"
    baseline_value = round(summary["baseline"][metric], 4)
    # one point per k, so the hover tooltip shows the baseline next to the curves at every k
    datasets.append({
        "label": "baseline (full document)",
        "data": [{"x": k, "y": baseline_value} for k in ks],
        "borderColor": baseline_color,
        "backgroundColor": baseline_color,
        "borderWidth": 2,
        "borderDash": [6, 4],
        "pointRadius": 0,
        "pointHoverRadius": 0,
    })

    if y_range is None:
        y_range = compute_y_range([d["y"] for dataset in datasets for d in dataset["data"]])
    y_min, y_max = y_range

    return f"""
    <div style="width: 360px;">
        <h4>{title}</h4>
        <canvas id="{chart_id}" width="360" height="320" style="width: 360px; height: 320px;"></canvas>
    </div>
    <script>
    new Chart(document.getElementById("{chart_id}"), {{
        type: 'line',
        data: {{
            datasets: {json.dumps(datasets)}
        }},
        options: {{
            responsive: false,
            interaction: {{ mode: 'x', intersect: false }},
            plugins: {{
                legend: {OVERVIEW_LEGEND_PLUGIN_OPTIONS}
            }},
            scales: {{
                y: {{ min: {y_min}, max: {y_max}, title: {{ display: true, text: {json.dumps(y_label)} }} }},
                x: {{
                    type: 'linear',
                    min: 1,
                    max: {max_k},
                    title: {{ display: true, text: 'k (tokens kept)' }}
                }}
            }}
        }}
    }});
    </script>
    """


def build_page(dataset_name, perturbation_name, model_mode_name, series, series_runs):
    documents = series_documents(series_runs)
    last = series_runs[-1]
    max_k = last["settings"]["max_k"]
    num_planned = len(last["plan"])
    summary = aggregate(documents, max_k)
    num_unchanged_at_max = summary["curves"][SETTINGS[0]][max_k - 1]["num_unchanged"]

    status = "" if len(documents) >= num_planned else f" - incomplete, {len(documents)}/{num_planned} planned"
    note = (
        f"{perturbation_name}, {model_mode_name}, series {series} ({len(series_runs)} runs): "
        f"{len(documents)} documents, {last['settings']['docs_per_author']} per author{status}. "
        f"Documents with k or fewer tokens stay whole at that k (at k = {max_k}: {num_unchanged_at_max} of {len(documents)})."
    )

    charts_html = ""
    for metric, title, y_label, y_range in METRICS:
        charts_html += build_keep_chart_html(
            chart_id=f"keep-{metric}", title=title, y_label=y_label,
            summary=summary, metric=metric, y_range=y_range, max_k=max_k,
        )

    return f"""<html>
<head>{PAGE_HEAD}</head>
<body>
<h1>keep top-k - {dataset_name}</h1>
{build_note_html(note)}
<div style="display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 40px;">
{charts_html}
</div>
</body>
</html>"""


def main():
    if not os.path.isdir(RESULTS_ROOT):
        print(f"no results under {RESULTS_ROOT}")
        return

    for dataset_name in sorted(os.listdir(RESULTS_ROOT)):
        dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue
        for perturbation_name in sorted(os.listdir(dataset_dir)):
            perturbation_dir = os.path.join(dataset_dir, perturbation_name)
            if not os.path.isdir(perturbation_dir):
                continue
            for model_mode_name in sorted(os.listdir(perturbation_dir)):
                series, series_runs = latest_series(load_runs(os.path.join(perturbation_dir, model_mode_name), "keep_"))
                if series is None:
                    continue

                output_dir = os.path.join(OUTPUTS_ROOT, dataset_name, perturbation_name, model_mode_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, "keep_top_k.html")
                with open(output_path, "w") as f:
                    f.write(build_page(dataset_name, perturbation_name, model_mode_name, series, series_runs))
                print(f"saved plots to {output_path}")


if __name__ == "__main__":
    main()
