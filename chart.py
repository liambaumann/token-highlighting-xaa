def build_chart_html(base_similarity, budget_results, title, link_href):
    labels = [0] + [budget["k"] for budget in budget_results]
    top_data = [base_similarity] + [budget["top_similarity"] for budget in budget_results]
    random_avg_data = [base_similarity] + [budget["random_similarity_avg"] for budget in budget_results]

    return f"""
    <a href="{link_href}" style="text-decoration:none; color:inherit;">
    <div style="width: 400px; margin-bottom: 40px; cursor:pointer;">
        <h3>{title}</h3>
        <canvas id="chart-{title}" width="400" height="300"></canvas>
    </div>
    </a>
    <script>
    new Chart(document.getElementById("chart-{title}"), {{
        type: 'line',
        data: {{
            labels: {labels},
            datasets: [
                {{ label: 'top-k removed', data: {[round(v, 4) for v in top_data]}, borderColor: 'red', backgroundColor: 'red' }},
                {{ label: 'random-k removed (n=5 avg)', data: {[round(v, 4) for v in random_avg_data]}, borderColor: 'blue', backgroundColor: 'blue' }}
            ]
        }},
        options: {{
            scales: {{
                y: {{ min: 0.7, max: 1.0 }},
                x: {{ title: {{ display: true, text: 'tokens removed (k)' }} }}
            }}
        }}
    }});
    </script>
    """
