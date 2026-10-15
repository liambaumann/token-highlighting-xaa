# One-off script: compare occlusion placeholder choices for masking (chunk32, n=1, darkreddit).
# Run from the repo root: python -m experiments.placeholder_test_darkreddit
# Does not modify any existing file; reuses model/data/scoring/masking as library code.

import os
import json
import time
import random
import torch

from model import model, tokenizer, device, embed_chunked
from data import load_authors, load_train_texts, load_test_texts
from scoring import tokenize
from masking import select_windows

random.seed(0)

RESULTS_DIR = "results/placeholder_test"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET = "darkreddit"
N = 1
BUDGET_PERCENTAGES = [0.01, 0.03, 0.05, 0.10]
NUM_RANDOM_DRAWS = 5
DOCS_PER_AUTHOR = 5
CORPUS_MEDIAN_SAMPLE_SIZE = 200_000
CORPUS_MEDIAN_SEED = 0
CENTROID_CACHE_PATH = "cache/darkreddit_luar-mud/centroids_chunk32.pt"

word_embeddings = model.transformer.embeddings.word_embeddings
HIDDEN_SIZE = word_embeddings.weight.shape[1]

BOS_ID = tokenizer.bos_token_id
EOS_ID = tokenizer.eos_token_id
PAD_ID = tokenizer.pad_token_id
MASK_ID = tokenizer.mask_token_id


# ---------- piece geometry (matches model.embed_variants' chunk32 branch exactly) ----------

def piece_ranges_for(num_tokens):
    content_end = num_tokens - 2  # last content position (num_tokens - 1 is </s>)
    num_pieces = max(1, -(-content_end // 30))  # ceil(content_end / 30)
    return [range(1 + 30 * j, min(30 + 30 * j, content_end) + 1) for j in range(num_pieces)]


# ---------- "delete" placeholder: today's behaviour, id-based, token removed ----------

def build_delete_piece(token_ids, piece_range, keep_set):
    kept = [p for p in piece_range if p in keep_set]
    piece = [BOS_ID] + [token_ids[p] for p in kept] + [EOS_ID]
    mask = [1] * len(piece)
    pad_len = 32 - len(piece)
    piece = piece + [PAD_ID] * pad_len
    mask = mask + [0] * pad_len
    return piece, mask


def embed_delete_variants(token_ids, piece_ranges, keep_lists, batch_size=32):
    embeddings = []
    for i in range(0, len(keep_lists), batch_size):
        batch_ids, batch_mask = [], []
        for keep in keep_lists[i:i+batch_size]:
            keep_set = set(keep)
            ids_pieces, mask_pieces = [], []
            for piece_range in piece_ranges:
                piece, mask = build_delete_piece(token_ids, piece_range, keep_set)
                ids_pieces.append(piece)
                mask_pieces.append(mask)
            batch_ids.append(ids_pieces)
            batch_mask.append(mask_pieces)

        ids_t = torch.tensor(batch_ids).to(device)
        mask_t = torch.tensor(batch_mask).to(device)
        with torch.no_grad():
            embeddings.append(model(input_ids=ids_t, attention_mask=mask_t).cpu())
    return torch.cat(embeddings, dim=0)


# ---------- placeholder-substitution path: fixed piece length, embeddings replaced ----------

def embed_episode_from_embeds(inputs_embeds, attention_mask):
    # inputs_embeds: (B, E, L, H); attention_mask: (B, E, L) - repeats LUAR.get_episode_embeddings
    # but starting from embeddings instead of input_ids
    B, E, L, H = inputs_embeds.shape
    flat_embeds = inputs_embeds.reshape(B * E, L, H)
    flat_mask = attention_mask.reshape(B * E, L)
    with torch.no_grad():
        out = model.transformer(
            inputs_embeds=flat_embeds, attention_mask=flat_mask,
            return_dict=True, output_hidden_states=False,
        )
        comment_embeddings = model.mean_pooling(out["last_hidden_state"], flat_mask)
        comment_embeddings = comment_embeddings.reshape(B, E, H)
        episode_embeddings = model.attn_fn(comment_embeddings, comment_embeddings, comment_embeddings)
        episode_embeddings = episode_embeddings.max(dim=1).values
        episode_embeddings = model.linear(episode_embeddings)
    return episode_embeddings.cpu()


def build_replace_piece_embeds(token_ids, piece_range, replace_set, placeholder_vec):
    bos_emb = word_embeddings(torch.tensor(BOS_ID, device=device))
    eos_emb = word_embeddings(torch.tensor(EOS_ID, device=device))
    pad_emb = word_embeddings(torch.tensor(PAD_ID, device=device))

    seq = [bos_emb]
    for p in piece_range:
        if p in replace_set:
            seq.append(placeholder_vec)
        else:
            seq.append(word_embeddings(torch.tensor(token_ids[p], device=device)))
    seq.append(eos_emb)
    mask = [1] * len(seq)
    pad_len = 32 - len(seq)
    seq = seq + [pad_emb] * pad_len
    mask = mask + [0] * pad_len
    return torch.stack(seq), torch.tensor(mask, device=device)


def embed_replace_variants(token_ids, piece_ranges, replace_sets, placeholder_vec, batch_size=32):
    embeddings = []
    for i in range(0, len(replace_sets), batch_size):
        batch_embeds, batch_mask = [], []
        for replace_set in replace_sets[i:i+batch_size]:
            piece_embeds, piece_masks = [], []
            for piece_range in piece_ranges:
                e, m = build_replace_piece_embeds(token_ids, piece_range, replace_set, placeholder_vec)
                piece_embeds.append(e)
                piece_masks.append(m)
            batch_embeds.append(torch.stack(piece_embeds))
            batch_mask.append(torch.stack(piece_masks))

        inputs_embeds = torch.stack(batch_embeds)
        attention_mask = torch.stack(batch_mask)
        embeddings.append(embed_episode_from_embeds(inputs_embeds, attention_mask))
    return torch.cat(embeddings, dim=0)


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b.unsqueeze(0), dim=1)


# ---------- corpus / vocab / doc statistics for the non-delete placeholders ----------

def compute_vocab_stats():
    weight = word_embeddings.weight.detach().cpu()
    return weight.mean(dim=0), weight.median(dim=0).values


def compute_corpus_stats(authors):
    print("collecting corpus content-token occurrences (Reuters train, all authors)...")
    occurrence_ids = []
    for author in authors:
        for text in load_train_texts(DATASET, author):
            ids = tokenizer(text, add_special_tokens=False)["input_ids"][:510]
            occurrence_ids.extend(ids)
    print(f"  {len(occurrence_ids)} content-token occurrences collected")

    occurrence_tensor = torch.tensor(occurrence_ids)
    vocab_size = word_embeddings.weight.shape[0]
    counts = torch.bincount(occurrence_tensor, minlength=vocab_size).float()
    weight = word_embeddings.weight.detach().cpu()
    corpus_mean = (counts.unsqueeze(1) * weight).sum(dim=0) / counts.sum()

    rng = random.Random(CORPUS_MEDIAN_SEED)
    sample_size = min(CORPUS_MEDIAN_SAMPLE_SIZE, len(occurrence_ids))
    sampled_ids = rng.sample(occurrence_ids, sample_size)
    sampled_embeds = weight[torch.tensor(sampled_ids)]
    corpus_median = sampled_embeds.median(dim=0).values

    return corpus_mean, corpus_median


def compute_doc_stats(token_ids, content_positions):
    embeds = word_embeddings(torch.tensor([token_ids[p] for p in content_positions], device=device)).detach().cpu()
    return embeds.mean(dim=0), embeds.median(dim=0).values


def closest_real_tokens(vec, k=10):
    weight = word_embeddings.weight.detach().cpu()
    sims = torch.nn.functional.cosine_similarity(weight, vec.unsqueeze(0), dim=1)
    top = sims.topk(k)
    tokens = tokenizer.convert_ids_to_tokens(top.indices.tolist())
    return [{"token": t, "cosine_similarity": round(s, 4)} for t, s in zip(tokens, top.values.tolist())]


# ---------- main per-document, per-placeholder computation ----------

def occlusion_scores_for(token_ids, piece_ranges, candidate_positions, base_similarity, centroid, placeholder_name, placeholder_vec):
    if placeholder_name == "delete":
        keep_lists = [
            [p for p in range(len(token_ids)) if p != pos]
            for pos in candidate_positions
        ]
        embeddings = embed_delete_variants(token_ids, piece_ranges, keep_lists)
    else:
        replace_sets = [{pos} for pos in candidate_positions]
        embeddings = embed_replace_variants(token_ids, piece_ranges, replace_sets, placeholder_vec)

    similarities = cos_sim(embeddings, centroid).tolist()
    scores = {pos: base_similarity - sim for pos, sim in zip(candidate_positions, similarities)}
    return scores


def evaluate_budgets_for(token_ids, piece_ranges, candidate_positions, scores, base_similarity, centroid, placeholder_name, placeholder_vec):
    sorted_desc = sorted(candidate_positions, key=lambda i: scores[i], reverse=True)
    sorted_asc = sorted(candidate_positions, key=lambda i: scores[i])

    budget_results = []
    for pct in BUDGET_PERCENTAGES:
        m = max(1, round(pct * len(candidate_positions)))

        top_starts = select_windows(sorted_desc, N, m)
        bottom_starts = select_windows(sorted_asc, N, m)
        random_starts_list = []
        for _ in range(NUM_RANDOM_DRAWS):
            shuffled = random.sample(candidate_positions, len(candidate_positions))
            random_starts_list.append(select_windows(shuffled, N, m))

        variant_sets = [set(top_starts)] + [set(rs) for rs in random_starts_list] + [set(bottom_starts)]

        if placeholder_name == "delete":
            keep_lists = [
                [p for p in range(len(token_ids)) if p not in variant_set]
                for variant_set in variant_sets
            ]
            embeddings = embed_delete_variants(token_ids, piece_ranges, keep_lists)
        else:
            embeddings = embed_replace_variants(token_ids, piece_ranges, variant_sets, placeholder_vec)

        similarities = cos_sim(embeddings, centroid).tolist()
        top_similarity = similarities[0]
        random_similarities = similarities[1:-1]
        bottom_similarity = similarities[-1]

        budget_results.append({
            "pct": pct,
            "k": m,
            "top_drop": base_similarity - top_similarity,
            "random_drop_avg": base_similarity - (sum(random_similarities) / len(random_similarities)),
            "bottom_drop": base_similarity - bottom_similarity,
        })

    return budget_results


def main():
    start_time = time.time()

    authors = load_authors(DATASET)
    # all 10 authors, first DOCS_PER_AUTHOR test comments each
    test_cases = [(author, i) for author in authors for i in range(DOCS_PER_AUTHOR)]

    centroids = torch.load(CENTROID_CACHE_PATH)

    vocab_mean, vocab_median = compute_vocab_stats()
    corpus_mean, corpus_median = compute_corpus_stats(authors)

    docs = []
    for author, doc_index in test_cases:
        text = load_test_texts(DATASET, author)[doc_index]
        input_ids, attention_mask = tokenize(text)
        token_ids = input_ids[0].tolist()
        num_tokens = input_ids.shape[1]
        piece_ranges = piece_ranges_for(num_tokens)
        candidate_positions = list(range(1, num_tokens - 1))

        # first check: nothing replaced must match embed_chunked within 1e-4
        check_embedding = embed_replace_variants(token_ids, piece_ranges, [set()], None)[0]
        reference_embedding = embed_chunked([text])
        diff = (check_embedding - reference_embedding).abs().max().item()
        print(f"[check] {author}: no-replacement path vs embed_chunked, max abs diff = {diff:.2e}")
        if diff >= 1e-4:
            raise SystemExit(f"STOPPING: baseline mismatch for {author} ({diff:.2e} >= 1e-4)")

        base_similarity = cos_sim(check_embedding.unsqueeze(0), centroids[author]).item()
        doc_mean, doc_median = compute_doc_stats(token_ids, candidate_positions)

        docs.append({
            "author": author,
            "doc_index": doc_index,
            "token_ids": token_ids,
            "num_tokens": num_tokens,
            "piece_ranges": piece_ranges,
            "candidate_positions": candidate_positions,
            "base_similarity": base_similarity,
            "doc_mean": doc_mean,
            "doc_median": doc_median,
        })

    placeholders = {
        "delete": None,
        "mask": word_embeddings(torch.tensor(MASK_ID, device=device)).detach(),
        "vocab_mean": vocab_mean.to(device),
        "vocab_median": vocab_median.to(device),
        "corpus_mean": corpus_mean.to(device),
        "corpus_median": corpus_median.to(device),
        # doc_mean / doc_median are per-document, filled in per doc below
    }

    results_by_placeholder = {}
    closest_tokens = {}

    for name in ["delete", "mask", "vocab_mean", "vocab_median", "corpus_mean", "corpus_median", "doc_mean", "doc_median"]:
        print(f"\n=== placeholder: {name} ===")
        per_doc_budgets = []

        for doc_idx, doc in enumerate(docs):
            if name in ("doc_mean", "doc_median"):
                placeholder_vec = doc["doc_mean"].to(device) if name == "doc_mean" else doc["doc_median"].to(device)
            else:
                placeholder_vec = placeholders[name]

            centroid = centroids[doc["author"]]
            scores = occlusion_scores_for(
                doc["token_ids"], doc["piece_ranges"], doc["candidate_positions"],
                doc["base_similarity"], centroid, name, placeholder_vec,
            )
            budget_results = evaluate_budgets_for(
                doc["token_ids"], doc["piece_ranges"], doc["candidate_positions"], scores,
                doc["base_similarity"], centroid, name, placeholder_vec,
            )
            per_doc_budgets.append(budget_results)
            print(f"  {doc['author']} doc {doc['doc_index']}: done")

            if doc_idx == 0 and name in ("doc_mean", "doc_median"):
                closest_tokens[name] = closest_real_tokens(placeholder_vec.cpu())

        # average drop per budget, across the 5 documents
        budget_summary = []
        for b in range(len(BUDGET_PERCENTAGES)):
            top_drops = [per_doc_budgets[d][b]["top_drop"] for d in range(len(docs))]
            random_drops = [per_doc_budgets[d][b]["random_drop_avg"] for d in range(len(docs))]
            bottom_drops = [per_doc_budgets[d][b]["bottom_drop"] for d in range(len(docs))]
            budget_summary.append({
                "pct": BUDGET_PERCENTAGES[b],
                "top_drop_avg": sum(top_drops) / len(top_drops),
                "random_drop_avg": sum(random_drops) / len(random_drops),
                "bottom_drop_avg": sum(bottom_drops) / len(bottom_drops),
            })

        aopc_top = sum(b["top_drop_avg"] for b in budget_summary) / len(budget_summary)
        aopc_random = sum(b["random_drop_avg"] for b in budget_summary) / len(budget_summary)
        aopc_bottom = sum(b["bottom_drop_avg"] for b in budget_summary) / len(budget_summary)

        results_by_placeholder[name] = {
            "budget_summary": budget_summary,
            "aopc_top": aopc_top,
            "aopc_random": aopc_random,
            "aopc_bottom": aopc_bottom,
            "gap": aopc_top - aopc_random,
        }

    runtime = time.time() - start_time

    # ---- print table ----
    print("\n\nplaceholder       | AOPC top | AOPC random | AOPC bottom | gap (top-random)")
    for name, r in results_by_placeholder.items():
        print(f"{name:<17} | {r['aopc_top']:+.4f}  | {r['aopc_random']:+.4f}     | {r['aopc_bottom']:+.4f}     | {r['gap']:+.4f}")

    print("\nclosest real tokens to doc_mean / doc_median (first document, user: "
          f"{docs[0]['author']}):")
    for name in ("doc_mean", "doc_median"):
        print(f"  {name}: {[t['token'] for t in closest_tokens[name]]}")

    print(f"\nruntime: {runtime:.1f}s")

    # ---- save json ----
    output = {
        "dataset": DATASET,
        "mode": "chunk32",
        "n": N,
        "budget_percentages": BUDGET_PERCENTAGES,
        "num_random_draws": NUM_RANDOM_DRAWS,
        "documents": [{"author": d["author"], "doc_index": d["doc_index"]} for d in docs],
        "results_by_placeholder": results_by_placeholder,
        "closest_real_tokens": closest_tokens,
        "runtime_seconds": runtime,
    }
    out_path = os.path.join(RESULTS_DIR, f"results_{DATASET}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"saved results to {out_path}")


if __name__ == "__main__":
    main()
