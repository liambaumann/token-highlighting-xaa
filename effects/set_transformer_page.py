# the page of effects.set_transformer, built from its result JSON only, nothing is computed (the
# display helpers come from effects.matching, whose import loads the model module).
# effects.set_transformer calls save_result_page after a run; run from the repo root to rebuild the
# page of a saved result: python -m effects.set_transformer_page   (--unit, default word; --n, only 1;
# --mean: zero-filled (default) or available-case; --file: a set_transformer_*.json in the mean's folder
# of the unit and n (unit_scores.result_dir), default the latest there). A rebuilt page gets a new
# timestamp, nothing is overwritten.
# Writes outputs/{dataset}/{perturbation}/{model_mode}/effects_set_transformer_{unit}_{mean}_{timestamp}.html.

import os
import json
import html
import argparse
from datetime import datetime
from runs import load_runs
from reporting.chart import VEGA_PAGE_HEAD, TOP_RGB, BOTTOM_RGB, build_note_html, build_vega_chart_html
from .units import UNIT_TYPES
from .unit_scores import (
    DAMPING_NOTE, MEANS, DEFAULT_MEAN, MEAN_NOTES, build_header_html, build_footer_html, save_page, result_dir,
)
from .matching import FREQUENCY_RGB, signature_table_rows, format_rate, hex_color

RESULT_PREFIX = "set_transformer_"
UNIT_VECTOR_LABELS = {"learned": "learned", "background-delta": "background Δ"}
MATCHING_LABELS = {"tf_svm": "TF-SVM", "delta_sum": "Δ-sum"}
WIDTH_COLORS = [TOP_RGB, BOTTOM_RGB, FREQUENCY_RGB]


def build_chart_spec(run, widths):
    # holdout top-1 per epoch (epoch 0 = before training), one line per width
    rows = [{"epoch": 0, "width": str(w["width"]), "holdout_top1": round(w["initial"]["holdout"]["top1"], 4)} for w in run["widths"]]
    rows += [{"epoch": e["epoch"], "width": str(w["width"]), "holdout_top1": round(e["holdout_top1"], 4)}
             for w in run["widths"] for e in w["epochs"]]
    best = [{"epoch": w["best_epoch"], "width": str(w["width"]), "holdout_top1": round(w["best_holdout_top1"], 4)} for w in run["widths"]]
    labels = [str(width) for width in widths]
    color = {"field": "width", "type": "nominal", "title": "width",
             "scale": {"domain": labels, "range": [hex_color(rgb) for rgb in WIDTH_COLORS[:len(widths)]]}}
    y = {"field": "holdout_top1", "type": "quantitative", "scale": {"domain": [0, 1]},
         "axis": {"format": ".0%", "title": "holdout top-1"}}
    x = {"field": "epoch", "type": "quantitative", "axis": {"title": "epoch", "tickMinStep": 1}}
    tooltip = [{"field": "width"}, {"field": "epoch"}, {"field": "holdout_top1", "title": "holdout top-1", "format": ".2%"}]
    return {
        "width": 400, "height": 240,
        "layer": [
            {"data": {"values": rows}, "mark": {"type": "line", "strokeWidth": 2},
             "encoding": {"x": x, "y": y, "color": color}},
            {"data": {"values": rows}, "mark": {"type": "point", "filled": True, "size": 14, "opacity": 0.6},
             "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip}},
            # the kept (best holdout) epoch of each width
            {"data": {"values": best}, "mark": {"type": "point", "size": 70, "strokeWidth": 2},
             "encoding": {"x": x, "y": y, "color": color, "tooltip": tooltip}},
        ],
    }


def build_table_html(result):
    numeric = 'padding: 2px 8px; text-align: right; font-variant-numeric: tabular-nums;'
    head_cell = 'padding: 2px 8px; font-weight: 600; border-bottom: 1px solid #ccc;'
    columns = ["chosen width", "holdout top-1", "test top-1"] + [f"{label} top-1" for label in MATCHING_LABELS.values()]
    head = (f'<th style="{head_cell} text-align: left;">signature</th><th style="{head_cell} text-align: left;">unit vectors</th>'
            + "".join(f'<th style="{head_cell} text-align: right;">{column}</th>' for column in columns))
    body = ""
    for key, label in result["settings"]["signature_types"].items():
        for i, (vectors, run) in enumerate(result["runs"][key].items()):
            cells = [str(run["chosen_width"]), format_rate(run["chosen_holdout_top1"]), format_rate(run["test"]["top1"])]
            cells += [format_rate(result["matching_k10"][key][method]) for method in MATCHING_LABELS]
            border = ' style="border-top: 1px solid #ddd;"' if i == 0 else ""
            body += (f'<tr{border}><td style="padding: 2px 8px;">{label if i == 0 else ""}</td>'
                     f'<td style="padding: 2px 8px;">{UNIT_VECTOR_LABELS[vectors]}</td>'
                     + "".join(f'<td style="{numeric}{" font-weight: 600;" if j == 2 else ""}">{cell}</td>' for j, cell in enumerate(cells))
                     + "</tr>")
    return f"""
<table style="border-collapse: collapse; font-size: 0.85em; margin-bottom: 16px;">
    <thead><tr>{head}</tr></thead>
    <tbody>{body}</tbody>
</table>"""


def result_mean(result):
    # results from before --mean have no "mean" and are zero-filled
    return result.get("mean", DEFAULT_MEAN)


def build_notes(result):
    s = result["settings"]
    optimizer = s["optimizer"]
    return [
        f"n = {result['ngram']}, k = {s['k']}; set = one element per pooled unit (union of all authors' top-k) in the comment: "
        f"unit vector + count / number of units; none of them: one learned empty element",
        "unit vectors: learned = embedding table at model width; background Δ = raw − corrected Δ of the unit (fixed), "
        "through a learned linear layer",
        f"model: {s['blocks']} self-attention blocks, {s['heads']} heads, feedforward {s['feedforward'].replace(' x ', ' × ')}, "
        f"attention pooling with one seed, linear to {s['output_dim']}; no positional information; dropout {s['dropout']}",
        f"training: {html.escape(s['loss'])}; {optimizer['name']} lr {optimizer['lr']}, weight decay {optimizer['weight_decay']}, "
        f"batch {s['batch_size']}, ≤ {s['max_epochs']} epochs, stop after {s['patience']} epochs without a better holdout top-1, "
        f"best holdout epoch kept; seed {s['seed']}",
        f"width {', '.join(map(str, s['widths']))} per run, the best holdout top-1 evaluated once on test; "
        f"TF-SVM and Δ-sum: k = {s['k']} from {html.escape(result['matching_file'])}",
        DAMPING_NOTE,
        MEAN_NOTES[result_mean(result)],
    ]


def build_page(result):
    data = result["data"]
    signature_types = result["settings"]["signature_types"]
    roles = (f"val {data['num_val_comments']} (signatures, Δ), train {data['num_train_comments']} (training), "
             f"train holdout {data['num_holdout_comments']} (stopping, width choice), test {data['num_test_comments']} "
             f"(evaluation only), top-1 on test")
    rows = [row for row in signature_table_rows(0) if row[0] in signature_types.values()]
    empty = "; ".join(
        f"{label}: train {result['empty_sets'][key]['train']}, holdout {result['empty_sets'][key]['holdout']}, "
        f"test {result['empty_sets'][key]['test']} ({result['num_pooled_units'][key]} pooled units)"
        for key, label in signature_types.items())
    charts = "".join(
        build_vega_chart_html(f"set-transformer-{key}-{vectors}", f"{label}, {UNIT_VECTOR_LABELS[vectors]}",
                              build_chart_spec(run, result["settings"]["widths"]), 480)
        for key, label in signature_types.items() for vectors, run in result["runs"][key].items())
    inputs = result["inputs"]
    names = [inputs["scoring_file"], inputs["deltas_file"], inputs["matching_file"], inputs["centroids_file"]]
    footer = [f"input {', '.join(html.escape(name) for name in names + inputs['cache_val_files'] + inputs['cache_test_files'])}; "
              f"cache series {html.escape(inputs['cache_series'])}; result {html.escape(result['file'])}"]
    return f"""<html>
<head>{VEGA_PAGE_HEAD}</head>
<body>
{build_header_html(f"set transformer - {html.escape(result['dataset'])}, {html.escape(result['unit'])} units", result, roles, rows,
                   build_notes(result))}
<h2>n = {result['ngram']}, k = {result['settings']['k']}</h2>
{build_table_html(result)}
{build_note_html("comments without any pooled unit (empty set): " + empty)}
{build_note_html("x: epoch (0 = before training), y: holdout top-1; large point: the kept epoch")}
<div style="display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 40px;">
{charts}
</div>
{build_footer_html(footer)}
</body>
</html>"""


def save_result_page(result, stamp):
    return save_page(f"effects_set_transformer_{result['unit']}_{result_mean(result)}", build_page(result), stamp)


def parse_args():
    parser = argparse.ArgumentParser(description="rebuild the page of a saved effects.set_transformer result")
    parser.add_argument("--unit", choices=sorted(UNIT_TYPES), default="word")
    parser.add_argument("--n", type=int, choices=[1], default=1)
    parser.add_argument("--mean", choices=MEANS, default=DEFAULT_MEAN)
    parser.add_argument("--file", help="a set_transformer_*.json in the mean's folder of the unit and n (default: the latest)")
    return parser.parse_args()


def main():
    args = parse_args()
    directory = result_dir(args.unit, args.n, args.mean)
    if args.file:
        with open(os.path.join(directory, args.file)) as f:
            result = json.load(f)
    else:
        runs = load_runs(directory, RESULT_PREFIX)
        if not runs:
            raise SystemExit(f"no {RESULT_PREFIX}*.json files in {directory} - run effects.set_transformer first")
        result = runs[-1]
    print(f"page of {result['file']}")
    save_result_page(result, f"{datetime.now():%Y%m%d-%H%M%S}")


if __name__ == "__main__":
    main()
