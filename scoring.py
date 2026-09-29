import torch
from model import embed_from_ids, tokenizer


def tokenize(text):
    text_no_newlines = " ".join(text.split())
    tokenized = tokenizer(text_no_newlines, max_length=512, truncation=True, return_tensors="pt")

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]
    return input_ids, attention_mask


def occlusion_scores(input_ids, attention_mask, centroid, n=1):
    base_similarity = torch.nn.functional.cosine_similarity(
        embed_from_ids(input_ids, attention_mask), centroid.unsqueeze(0), dim=1
    ).item()

    num_tokens = input_ids.shape[1]
    num_windows = num_tokens - n + 1

    # windows that would remove position 0 (<s>) or num_tokens - 1 (</s>) are skipped
    valid_starts = list(range(1, num_tokens - n))

    loo_ids = torch.cat([
        torch.cat([input_ids[:, :i], input_ids[:, i+n:]], dim=1)
        for i in valid_starts
    ], dim=0)
    loo_mask = torch.cat([
        torch.cat([attention_mask[:, :i], attention_mask[:, i+n:]], dim=1)
        for i in valid_starts
    ], dim=0)
    loo_embeddings = embed_from_ids(loo_ids, loo_mask)
    perturbed_similarities = torch.nn.functional.cosine_similarity(
        loo_embeddings, centroid.unsqueeze(0), dim=1
    ).tolist()

    scores = [0.0] * num_windows
    for idx, i in enumerate(valid_starts):
        scores[i] = base_similarity - perturbed_similarities[idx]

    return base_similarity, scores
