import os
import json
import config
from model import tokenizer
from highlight import save_highlighted_html
from chart import build_chart_html

results = []
for fname in os.listdir(config.RESULTS_DIR):
    if not fname.endswith(".json"):
        continue
    with open(os.path.join(config.RESULTS_DIR, fname)) as f:
        results.append(json.load(f))

results.sort(key=lambda r: (r["author"], r["doc_index"], r["n"]))

all_charts_html = ""

for result in results:
    author = result["author"]
    doc_index = result["doc_index"]
    n = result["n"]

    filename = f"{author}_{doc_index}"
    if n == 1:
        highlight_path = save_highlighted_html(result["tokens"], result["scores"], filename, tokenizer)
    else:
        highlight_path = os.path.join("outputs", f"{filename}.html")

    chart_title = f"{author}_{doc_index}_n{n}"
    all_charts_html += build_chart_html(
        result["base_similarity"], result["budget_results"],
        title=chart_title, link_href=os.path.basename(highlight_path),
    )

overview_html = f"""<html>
<head><script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script></head>
<body>
{all_charts_html}
</body>
</html>"""

with open("outputs/masking_reuters_overview.html", "w") as f:
    f.write(overview_html)

print("\nsaved overview to outputs/masking_reuters_overview.html")
