import torch
from model import embed_variants, tokenizer


def tokenize(text):
    tokenized = tokenizer(text, max_length=512, truncation=True, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    return input_ids, attention_mask


def occlusion_scores(input_ids, attention_mask, centroid, n=1):
    num_tokens = input_ids.shape[1]
    all_positions = list(range(num_tokens))

    base_embedding = embed_variants(input_ids, attention_mask, [all_positions])
    base_similarity = torch.nn.functional.cosine_similarity(
        base_embedding, centroid.unsqueeze(0), dim=1
    ).item()

    num_windows = num_tokens - n + 1

    # windows that would remove position 0 (<s>) or num_tokens - 1 (</s>) are skipped
    valid_starts = list(range(1, num_tokens - n))

    keep_lists = [
        [i for i in all_positions if not (start <= i < start + n)]
        for start in valid_starts
    ]
    loo_embeddings = embed_variants(input_ids, attention_mask, keep_lists)
    perturbed_similarities = torch.nn.functional.cosine_similarity(
        loo_embeddings, centroid.unsqueeze(0), dim=1
    ).tolist()

    scores = [0.0] * num_windows
    for idx, i in enumerate(valid_starts):
        scores[i] = base_similarity - perturbed_similarities[idx]

    return base_similarity, scores
