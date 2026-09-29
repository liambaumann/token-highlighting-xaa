import random
import torch
from model import embed_from_ids


def select_windows(order, n, m):
    selected = []
    occupied = set()
    for start in order:
        window = range(start, start + n)
        if any(p in occupied for p in window):
            continue
        selected.append(start)
        occupied.update(window)
        if len(selected) == m:
            break
    return selected


def evaluate_budgets(input_ids, attention_mask, centroid, scores, budget_percentages, n=1):
    num_tokens = input_ids.shape[1]
    candidate_positions = list(range(1, num_tokens - 1))
    candidate_starts = list(range(1, num_tokens - n))
    sorted_starts = sorted(candidate_starts, key=lambda i: scores[i], reverse=True)

    budget_results = []
    for pct in budget_percentages:
        target_k = max(1, round(pct * len(candidate_positions)))
        m = max(1, round(target_k / n))

        top_starts = select_windows(sorted_starts, n, m)
        top_indices = [p for start in top_starts for p in range(start, start + n)]
        assert all(0 < i < num_tokens - 1 for i in top_indices)
        keep_masks = [[i for i in range(input_ids.shape[1]) if i not in top_indices]]

        for _ in range(5):
            shuffled_starts = random.sample(candidate_starts, len(candidate_starts))
            random_starts = select_windows(shuffled_starts, n, m)
            random_indices = [p for start in random_starts for p in range(start, start + n)]
            assert all(0 < i < num_tokens - 1 for i in random_indices)
            keep_masks.append([i for i in range(input_ids.shape[1]) if i not in random_indices])

        batch_ids = torch.cat([input_ids[:, keep_mask] for keep_mask in keep_masks], dim=0)
        batch_attention = torch.cat([attention_mask[:, keep_mask] for keep_mask in keep_masks], dim=0)
        batch_embeddings = embed_from_ids(batch_ids, batch_attention)
        similarities = torch.nn.functional.cosine_similarity(
            batch_embeddings, centroid.unsqueeze(0), dim=1
        ).tolist()

        top_similarity = similarities[0]
        random_similarities = similarities[1:]

        budget_results.append({
            "k": m * n,
            "n": n,
            "pct": pct,
            "top_similarity": top_similarity,
            "random_similarities": random_similarities,
            "random_similarity_avg": sum(random_similarities) / len(random_similarities),
        })

    return budget_results
