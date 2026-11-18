"""Test top-1 over k for TF-SVM, Δ-sum and the set transformer (learned unit vectors).

Numbers are copied from the "Main comparison: test top-1" table in results/misc/set_transformer.md
(zero-filled mean only) and the full-LUAR ceiling from its "Reference" table.

Run from anywhere:  venv/bin/python results/misc/graphs/set_transformer/graph.py
Writes two versions to outputs/, each as .svg (for \\includesvg) and .vl.json (the spec):
- set_transformer:        one chart, color = method, dash = signature, legend below
- set_transformer_split:  one chart per signature side by side, labels at the line ends
"""

import json
import sys
from pathlib import Path

import vl_convert as vlc

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # repo root, for reporting.chart

from reporting.chart import VEGA_CONFIG, vega_spec  # noqa: E402

OUT_DIR = HERE / "outputs"

FONT = "Latin Modern Roman"

# (k, signature, TF-SVM, Δ-sum, ST learned) - zero-filled rows of the main comparison table
ROWS = [
    (5, "Euclidean distance", 50.8, 62.2, 62.0),
    (5, "relative frequency", 60.1, 65.5, 67.1),
    (10, "Euclidean distance", 59.4, 64.0, 68.1),
    (10, "relative frequency", 64.9, 66.0, 71.6),
    (20, "Euclidean distance", 62.4, 60.0, 73.5),
    (20, "relative frequency", 67.3, 65.4, 73.2),
]
FULL_LUAR = 88.1

METHODS = ["TF-SVM", "Δ-sum", "ST learned"]
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs)
SIGNATURES = ["Euclidean distance", "relative frequency"]
Y_SCALE = {"domain": [50, 90], "zero": False}  # on every layer, else the layers' union starts at 0
X_SCALE = {"type": "log", "domain": [5, 20], "nice": False, "padding": 14}
X_AXIS = {"values": [5, 10, 20], "format": "d", "grid": False}
Y_TITLE = "test top-1 (%)"
TEXT_COLOR = "#52514e"


def long_rows():
    out = []
    for k, signature, *scores in ROWS:
        for method, top1 in zip(METHODS, scores):
            out.append({"k": k, "signature": signature, "method": method, "top1": top1})
    return out


def x_enc():
    return {"field": "k", "type": "quantitative", "title": "k", "scale": X_SCALE, "axis": X_AXIS}


def color_enc(legend=True):
    enc = {"field": "method", "type": "nominal", "title": None,
           "scale": {"domain": METHODS, "range": METHOD_COLORS}}
    if not legend:
        enc["legend"] = None
    return enc


def ceiling_layers():
    rule = {
        "data": {"values": [{"top1": FULL_LUAR}]},
        "mark": {"type": "rule", "color": "#8a8a85", "strokeWidth": 1.5, "strokeDash": [6, 4]},
        "encoding": {"y": {"field": "top1", "type": "quantitative", "scale": Y_SCALE}},
    }
    label = {
        "data": {"values": [{"top1": FULL_LUAR, "k": 5, "label": f"full LUAR (ceiling) {FULL_LUAR}"}]},
        "mark": {"type": "text", "align": "left", "baseline": "top", "dx": 0, "dy": 4,
                 "color": TEXT_COLOR},
        "encoding": {"x": {"field": "k", "type": "quantitative", "scale": X_SCALE},
                     "y": {"field": "top1", "type": "quantitative", "scale": Y_SCALE},
                     "text": {"field": "label"}},
    }
    return [rule, label]


LINE_MARK = {"type": "line", "strokeWidth": 2,
             "point": {"filled": True, "size": 40, "stroke": "white", "strokeWidth": 1}}
TOOLTIP = [{"field": "method"}, {"field": "signature"}, {"field": "k"},
           {"field": "top1", "title": "test top-1"}]


def build_spec():
    lines = {
        "data": {"values": long_rows()},
        "encoding": {
            "x": x_enc(),
            "y": {"field": "top1", "type": "quantitative", "title": Y_TITLE,
                  "scale": Y_SCALE, "axis": {"tickCount": 5}},
            "color": color_enc(),
            "strokeDash": {"field": "signature", "type": "nominal", "title": None,
                           "scale": {"domain": SIGNATURES, "range": [[1, 0], [2, 3]]},
                           "legend": {"encode": {"symbols": {"update": {"stroke": {"value": TEXT_COLOR}}}}}},
            "detail": {"field": "signature"},
            "tooltip": TOOLTIP,
        },
        "mark": LINE_MARK,
    }
    return vega_spec({
        "width": 440,
        "height": 260,
        "layer": [*ceiling_layers(), lines],
    })


SPLIT_WIDTH = 175
SPLIT_HEIGHT = 260
LABEL_GAP_PX = 14  # min vertical distance between two end labels


def spread_labels(values, min_gap):
    # nudge label positions apart (equally, both directions) until neighbours are min_gap apart
    order = sorted(range(len(values)), key=lambda i: values[i])
    pos = [values[i] for i in order]
    for _ in range(50):
        moved = False
        for a in range(len(pos) - 1):
            overlap = min_gap - (pos[a + 1] - pos[a])
            if overlap > 1e-9:
                pos[a] -= overlap / 2
                pos[a + 1] += overlap / 2
                moved = True
        if not moved:
            break
    out = [0.0] * len(values)
    for rank, i in enumerate(order):
        out[i] = pos[rank]
    return out


def split_panel(signature, first):
    rows = [r for r in long_rows() if r["signature"] == signature]
    last_k = max(r["k"] for r in rows)
    ends = [r for r in rows if r["k"] == last_k]
    y_domain = Y_SCALE["domain"]
    min_gap = LABEL_GAP_PX * (y_domain[1] - y_domain[0]) / SPLIT_HEIGHT
    for r, label_y in zip(ends, spread_labels([r["top1"] for r in ends], min_gap)):
        r["label_y"] = label_y

    # right panel: same ticks, labels only invisible (not removed) so both panels have the same
    # bounds and the titles line up
    y_axis = {"tickCount": 5} if first else {"tickCount": 5, "labelOpacity": 0, "ticks": False,
                                             "title": None}
    lines = {
        "data": {"values": rows},
        "encoding": {
            "x": x_enc(),
            "y": {"field": "top1", "type": "quantitative", "title": Y_TITLE, "scale": Y_SCALE,
                  "axis": y_axis},
            "color": color_enc(legend=False),
            "tooltip": TOOLTIP,
        },
        "mark": LINE_MARK,
    }
    end_labels = {
        "data": {"values": ends},
        "mark": {"type": "text", "align": "left", "baseline": "middle", "dx": 9,
                 "color": TEXT_COLOR},
        "encoding": {
            "x": {"field": "k", "type": "quantitative", "scale": X_SCALE},
            "y": {"field": "label_y", "type": "quantitative", "scale": Y_SCALE},
            "text": {"field": "method"},
        },
    }
    return {
        "title": signature,
        "width": SPLIT_WIDTH,
        "height": SPLIT_HEIGHT,
        "layer": [*ceiling_layers(), lines, end_labels],
    }


def build_split_spec():
    return vega_spec({
        "hconcat": [split_panel(sig, i == 0) for i, sig in enumerate(SIGNATURES)],
        "spacing": 60,  # room for the left panel's end labels
    })


def thesis_config():
    # shared chart look, but Latin Modern and sizes that read as ~10pt when the SVG is included at
    # its natural size (1px = 0.75pt)
    config = json.loads(json.dumps(VEGA_CONFIG))
    config["font"] = FONT
    config["axis"].update({"labelFontSize": 12, "titleFontSize": 13, "titleFontWeight": "normal"})
    config["legend"].update({"labelFontSize": 12, "symbolStrokeWidth": 2, "symbolSize": 500,
                             "direction": "horizontal", "columnPadding": 16})
    config["text"] = {"font": FONT, "fontSize": 12}
    config["title"] = {"font": FONT, "fontSize": 13, "fontWeight": "normal", "color": "#1a1a1a",
                       "offset": 8}
    config["background"] = "white"
    return config


def write(name, spec):
    spec["config"] = thesis_config()
    (OUT_DIR / f"{name}.vl.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    (OUT_DIR / f"{name}.svg").write_text(vlc.vegalite_to_svg(spec))
    print(f"wrote {OUT_DIR / name}.svg")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    write("set_transformer", build_spec())
    write("set_transformer_split", build_split_spec())


if __name__ == "__main__":
    main()
