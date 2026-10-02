import os
import json
from model import tokenizer
from highlight import save_highlighted_html
from chart import build_chart_html

RESULTS_ROOT = "results/masking"
OUTPUTS_ROOT = "outputs"

for dataset_name in sorted(os.listdir(RESULTS_ROOT)):
    dataset_dir = os.path.join(RESULTS_ROOT, dataset_name)
    if not os.path.isdir(dataset_dir):
        continue

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

        all_charts_html = ""

        for result in results:
            author = result["author"]
            doc_index = result["doc_index"]
            n = result["n"]

            filename = f"{author}_{doc_index}"
            if n == 1:
                highlight_path = save_highlighted_html(result["tokens"], result["scores"], filename, tokenizer, output_dir=output_dir)
            else:
                highlight_path = os.path.join(output_dir, f"{filename}.html")

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

        overview_path = os.path.join(output_dir, "masking_reuters_overview.html")
        with open(overview_path, "w") as f:
            f.write(overview_html)

        print(f"\nsaved overview to {overview_path}")
