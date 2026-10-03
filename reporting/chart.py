import json

FONT_STACK = "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

PAGE_HEAD = f"""
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
    body {{ font-family: {FONT_STACK}; color: #1a1a1a; }}
    h1, h2, h3, h4 {{ font-weight: 600; }}
</style>
<script>
    Chart.defaults.animation = false;
    Chart.defaults.font.family = {json.dumps(FONT_STACK)};
</script>
"""

# plain small-box legend (no usePointStyle - that combo has sizing quirks across Chart.js
# versions); hides the band datasets (empty label) so only the three mean/value lines show
OVERVIEW_LEGEND_PLUGIN_OPTIONS = """{
                position: 'bottom',
                labels: {
                    boxWidth: 10, boxHeight: 10, padding: 12, font: { size: 11 },
                    filter: (item) => item.text !== ''
                }
            }"""

# shared RGB palette for top/random/bottom, used by both the per-document and overview charts
TOP_RGB = (42, 120, 214)
RANDOM_RGB = (138, 138, 133)
BOTTOM_RGB = (235, 104, 52)


def build_note_html(note):
    return f'<div style="color:#666; font-size:0.9em; margin-top:-8px; margin-bottom:24px;">{note}</div>'




def percentile(values, pct):
    # linear-interpolation percentile (numpy's default method), no numpy dependency
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    rank = pct / 100 * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_vals[lower] + (sorted_vals[upper] - sorted_vals[lower]) * frac


# (key into a per-document "drops" dict, RGB for color, legend label) - kept short so 3 fit
# on one legend row at 360px wide
OVERVIEW_SELECTIONS = [
    ("top", TOP_RGB, "top-k"),
    ("random_avg", RANDOM_RGB, "random-k (avg)"),
    ("bottom", BOTTOM_RGB, "bottom-k"),
]


def build_overview_chart_html(chart_id, title, docs, x_labels, y_range, link_href=None):
    # docs: list of {"top": [...], "random_avg": [...], "bottom": [...]}, each list aligned to x_labels
    datasets = []

    for key, rgb, label in OVERVIEW_SELECTIONS:
        band_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, 0.2)"
        solid_color = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"

        p25 = []
        p75 = []
        mean = []
        for i in range(len(x_labels)):
            values = [doc[key][i] for doc in docs]
            p25.append(percentile(values, 25))
            p75.append(percentile(values, 75))
            mean.append(sum(values) / len(values))

        upper_points = [{"x": x, "y": round(v, 4)} for x, v in zip(x_labels, p75)]
        lower_points = [{"x": x, "y": round(v, 4)} for x, v in zip(x_labels, p25)]
        mean_points = [{"x": x, "y": round(v, 4)} for x, v in zip(x_labels, mean)]

        datasets.append({
            "label": "",
            "data": upper_points,
            "borderColor": "transparent",
            "backgroundColor": band_color,
            "borderWidth": 0,
            "pointRadius": 0,
            "fill": False,
        })
        datasets.append({
            "label": "",
            "data": lower_points,
            "borderColor": "transparent",
            "backgroundColor": band_color,
            "borderWidth": 0,
            "pointRadius": 0,
            "fill": "-1",
        })
        datasets.append({
            "label": label,
            "data": mean_points,
            "borderColor": solid_color,
            "backgroundColor": solid_color,
            "borderWidth": 3,
            "pointRadius": 4,
        })

    y_min, y_max = y_range

    chart_div = f"""
    <div style="width: 360px; {'cursor:pointer;' if link_href else ''}">
        <h4>{title}</h4>
        <canvas id="{chart_id}" width="360" height="320" style="width: 360px; height: 320px;"></canvas>
    </div>
    """
    if link_href:
        chart_div = f'<a href="{link_href}" style="text-decoration:none; color:inherit;">{chart_div}</a>'

    return f"""
    {chart_div}
    <script>
    new Chart(document.getElementById("{chart_id}"), {{
        type: 'line',
        data: {{
            datasets: {json.dumps(datasets)}
        }},
        options: {{
            responsive: false,
            plugins: {{
                legend: {OVERVIEW_LEGEND_PLUGIN_OPTIONS}
            }},
            scales: {{
                y: {{ min: {y_min}, max: {y_max}, title: {{ display: true, text: 'similarity drop' }} }},
                x: {{
                    type: 'linear',
                    min: 0,
                    max: 10,
                    afterBuildTicks: (axis) => {{ axis.ticks = [0, 1, 3, 5, 10].map((v) => ({{ value: v }})); }},
                    title: {{ display: true, text: 'tokens removed (% of document)' }}
                }}
            }}
        }}
    }});
    </script>
    """


def no_results_chart_html(title):
    return f"""
    <div style="width: 360px; height: 320px; display: flex; flex-direction: column;
                align-items: center; justify-content: center; border: 1px dashed #999; color: #999;">
        <h4 style="color: #999;">{title}</h4>
        <div>no results</div>
    </div>
    """
