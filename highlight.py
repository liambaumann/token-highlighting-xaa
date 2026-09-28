import os
from datetime import datetime


def save_highlighted_html(tokens, scores, filename, tokenizer, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    max_abs_score = max(abs(s) for s in scores) or 1.0

    html_parts = []
    for token, score in zip(tokens, scores):
        if token in ("<s>", "</s>"):
            continue
        clean_token = tokenizer.convert_tokens_to_string([token])
        starts_new_word = token.startswith("Ġ") or token.startswith("Ċ")
        intensity = min(abs(score) / max_abs_score, 1.0)
        color = f"rgba(255,0,0,{intensity:.2f})" if score > 0 else f"rgba(0,0,255,{intensity:.2f})"
        css_class = "tok-word" if starts_new_word else "tok-piece"
        span = (
            f'<span class="{css_class}" style="background-color:{color};" '
            f'data-score="{score:+.4f}">{clean_token}</span>'
        )
        html_parts.append(span)

    body = "<!--\n-->".join(html_parts)

    html = f"""<html>
<head>
<style>
.tok-word, .tok-piece {{
    position: relative;
}}
.tok-piece::before {{
    content: "";
    position: absolute;
    left: 0;
    top: -3px;
    bottom: -3px;
    width: 1px;
    background: #555;
}}
.tok-word:hover::after, .tok-piece:hover::after {{
    content: attr(data-score);
    position: absolute;
    bottom: 100%;
    left: 0;
    background: #333;
    color: white;
    padding: 2px 6px;
    font-size: 12px;
    border-radius: 3px;
    white-space: nowrap;
    z-index: 10;
}}
</style>
</head>
<body style="font-family:sans-serif; line-height:1.8; padding:2em;">
{body}
</body>
</html>
"""

    filepath = os.path.join(output_dir, f"{filename}.html")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html)

    return filepath