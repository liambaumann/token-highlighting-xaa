# loads LUAR model into models/ next to this file (downloaded on first run)

import os

# must be set before transformers is imported
# all Hugging Face files (model, custom code, caches) go to models/ instead of ~/.cache/huggingface
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
# Triton (used by PyTorch on GPU) caches compiled kernels in ~/.triton by default
os.environ["TRITON_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")

from transformers import AutoTokenizer, AutoModel
import torch
import config

# use GPU if available (on VSC)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

MODELS = {"luar-mud": "rrivera1849/LUAR-MUD"}

def finetuned_path(dataset, run_name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "finetuned", dataset, run_name, "best", "model.pt")

# fine-tuned models: base LUAR-MUD with the best state_dict saved by finetuning/trainer.py loaded on top
# (tokenizer is always the base one). Add a real run with one line, e.g.
#     "luar-mud-ft-reuters-top2": finetuned_path("reuters", "top2_lr2e-05"),
FINETUNED_MODELS = {
    "luar-mud-ft-reuters-smoke": finetuned_path("reuters", "top2_lr2e-05_smoke"),
    "luar-mud-ft-darkreddit": finetuned_path("darkreddit", "top2_lr2e-05"),
    "luar-mud-ft-darkreddit-top4": finetuned_path("darkreddit", "top4_lr5e-05"),
}

def load_model(name):
    if name in MODELS:
        return AutoModel.from_pretrained(MODELS[name], trust_remote_code=True)
    if name not in FINETUNED_MODELS:
        raise ValueError(f"unknown model {name!r}, known: {sorted(MODELS) + sorted(FINETUNED_MODELS)}")
    path = FINETUNED_MODELS[name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"fine-tuned model {name!r}: {path} does not exist")
    finetuned = AutoModel.from_pretrained(MODELS["luar-mud"], trust_remote_code=True)
    finetuned.load_state_dict(torch.load(path, map_location="cpu"))
    return finetuned

tokenizer = AutoTokenizer.from_pretrained(MODELS["luar-mud"], trust_remote_code=True)
model = load_model(config.MODEL)
model.eval()
model.to(device)

word_embeddings = model.transformer.embeddings.word_embeddings

def embed(text, max_length=512):
    tokenized = tokenizer(
        [text],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized["input_ids"] = tokenized["input_ids"].reshape(1, 1, -1).to(device)
    tokenized["attention_mask"] = tokenized["attention_mask"].reshape(1, 1, -1).to(device)
    with torch.no_grad():
        return model(**tokenized).squeeze(0).cpu()

def embed_episode(texts, max_length=512):
    episode_length = len(texts)
    tokenized = tokenizer(
        texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized["input_ids"] = tokenized["input_ids"].reshape(1, episode_length, -1).to(device)
    tokenized["attention_mask"] = tokenized["attention_mask"].reshape(1, episode_length, -1).to(device)
    with torch.no_grad():
        return model(**tokenized).squeeze(0).cpu()

def chunk_pieces(text):
    # first 510 content tokens of text split into 30-token pieces, each wrapped in <s></s>
    # and padded to 32; returns (list of piece ids, list of piece masks)
    pieces_ids = []
    pieces_mask = []
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:510]
    for i in range(0, len(ids), 30):
        piece = [tokenizer.bos_token_id] + ids[i:i+30] + [tokenizer.eos_token_id]
        mask = [1] * len(piece)
        pad_len = 32 - len(piece)
        piece = piece + [tokenizer.pad_token_id] * pad_len
        mask = mask + [0] * pad_len
        pieces_ids.append(piece)
        pieces_mask.append(mask)
    return pieces_ids, pieces_mask

def embed_chunked(texts):
    # tokenize each text into 30-token pieces (plus <s>/</s>, padded to 32), all pieces from
    # all texts form one episode, so the model aggregates across pieces the same way it
    # aggregates across texts in embed_episode
    all_pieces_ids = []
    all_pieces_mask = []
    for text in texts:
        pieces_ids, pieces_mask = chunk_pieces(text)
        all_pieces_ids.extend(pieces_ids)
        all_pieces_mask.extend(pieces_mask)

    num_pieces = len(all_pieces_ids)
    input_ids = torch.tensor(all_pieces_ids).reshape(1, num_pieces, 32).to(device)
    attention_mask = torch.tensor(all_pieces_mask).reshape(1, num_pieces, 32).to(device)
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).cpu()

def content_token_count(text, max_tokens=510):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens])

def embed_single(texts, starts):
    # one 30-content-token excerpt per text (positions starts[i] .. starts[i]+30 of that
    # text's first 510 content tokens), wrapped in <s></s>, as one episode of length len(texts)
    all_ids = []
    all_mask = []
    for text, start in zip(texts, starts):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:510]
        piece = [tokenizer.bos_token_id] + ids[start:start+30] + [tokenizer.eos_token_id]
        mask = [1] * len(piece)
        pad_len = 32 - len(piece)
        piece = piece + [tokenizer.pad_token_id] * pad_len
        mask = mask + [0] * pad_len
        all_ids.append(piece)
        all_mask.append(mask)

    episode_length = len(texts)
    input_ids = torch.tensor(all_ids).reshape(1, episode_length, 32).to(device)
    attention_mask = torch.tensor(all_mask).reshape(1, episode_length, 32).to(device)
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask).squeeze(0).cpu()

def _piece_ranges_for(num_tokens):
    content_end = num_tokens - 2  # last content position (num_tokens - 1 is </s>)
    num_pieces = max(1, -(-content_end // 30))  # ceil(content_end / 30)
    return [range(1 + 30 * j, min(30 + 30 * j, content_end) + 1) for j in range(num_pieces)]


def embed_variants(input_ids, attention_mask, keep_lists):
    # input_ids, attention_mask: shape (1, num_tokens), a single tokenized document (<s>, content, </s>)
    # keep_lists: list of variants, each a sorted list of positions (into input_ids) to keep -
    # positions not in a keep list are the "dropped" positions, filled in per config.PERTURBATION
    if config.PERTURBATION == "mask":
        return embed_variants_mask(input_ids, attention_mask, keep_lists)
    elif config.PERTURBATION == "doc_mean":
        return embed_variants_doc_mean(input_ids, attention_mask, keep_lists)
    elif config.PERTURBATION == "delete":
        return embed_variants_delete(input_ids, attention_mask, keep_lists)
    else:
        raise ValueError(f"unknown config.PERTURBATION: {config.PERTURBATION!r}")


def embed_variants_delete(input_ids, attention_mask, keep_lists):
    # today's behaviour: dropped positions are removed, pieces/sequences shrink
    if config.MODE != "chunk32":
        variant_ids = torch.cat([input_ids[:, keep] for keep in keep_lists], dim=0)
        variant_mask = torch.cat([attention_mask[:, keep] for keep in keep_lists], dim=0)
        return embed_from_ids(variant_ids, variant_mask)

    num_tokens = input_ids.shape[1]
    piece_ranges = _piece_ranges_for(num_tokens)

    token_ids = input_ids[0].tolist()
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    all_variants_ids = []
    all_variants_mask = []
    for keep in keep_lists:
        keep_set = set(keep)
        variant_ids = []
        variant_mask = []
        for piece_range in piece_ranges:
            kept_in_piece = [p for p in piece_range if p in keep_set]
            piece = [bos_id] + [token_ids[p] for p in kept_in_piece] + [eos_id]
            mask = [1] * len(piece)
            pad_len = 32 - len(piece)
            piece = piece + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            variant_ids.append(piece)
            variant_mask.append(mask)
        all_variants_ids.append(variant_ids)
        all_variants_mask.append(variant_mask)

    variants_ids = torch.tensor(all_variants_ids)
    variants_mask = torch.tensor(all_variants_mask)

    embeddings = []
    for i in range(0, variants_ids.shape[0], BATCH_SIZE):
        chunk_ids = variants_ids[i:i+BATCH_SIZE].to(device)
        chunk_mask = variants_mask[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            embeddings.append(model(input_ids=chunk_ids, attention_mask=chunk_mask).cpu())
    return torch.cat(embeddings, dim=0)


def embed_variants_mask(input_ids, attention_mask, keep_lists):
    # dropped positions keep their place; token id replaced with the mask token. <s>, </s>
    # and padding are never dropped (masking.py never includes them in a candidate position).
    mask_id = tokenizer.mask_token_id
    num_tokens = input_ids.shape[1]
    token_ids = input_ids[0].tolist()

    if config.MODE != "chunk32":
        variant_ids = []
        for keep in keep_lists:
            keep_set = set(keep)
            variant_ids.append([token_ids[p] if p in keep_set else mask_id for p in range(num_tokens)])
        variant_ids = torch.tensor(variant_ids)
        variant_mask = attention_mask.repeat(len(keep_lists), 1)
        return embed_from_ids(variant_ids, variant_mask)

    piece_ranges = _piece_ranges_for(num_tokens)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    all_variants_ids = []
    all_variants_mask = []
    for keep in keep_lists:
        keep_set = set(keep)
        variant_ids = []
        variant_mask = []
        for piece_range in piece_ranges:
            piece_content = [token_ids[p] if p in keep_set else mask_id for p in piece_range]
            piece = [bos_id] + piece_content + [eos_id]
            mask = [1] * len(piece)
            pad_len = 32 - len(piece)
            piece = piece + [pad_id] * pad_len
            mask = mask + [0] * pad_len
            variant_ids.append(piece)
            variant_mask.append(mask)
        all_variants_ids.append(variant_ids)
        all_variants_mask.append(variant_mask)

    variants_ids = torch.tensor(all_variants_ids)
    variants_mask = torch.tensor(all_variants_mask)

    embeddings = []
    for i in range(0, variants_ids.shape[0], BATCH_SIZE):
        chunk_ids = variants_ids[i:i+BATCH_SIZE].to(device)
        chunk_mask = variants_mask[i:i+BATCH_SIZE].to(device)
        with torch.no_grad():
            embeddings.append(model(input_ids=chunk_ids, attention_mask=chunk_mask).cpu())
    return torch.cat(embeddings, dim=0)


def embed_variants_doc_mean(input_ids, attention_mask, keep_lists):
    # dropped positions keep their place; word embedding replaced with the mean word embedding
    # of this document's own (unperturbed) content tokens. Needs embeddings, not ids, so LUAR
    # is run from inputs_embeds the same way experiments/placeholder_test_reuters.py does.
    from experiments.placeholder_test_reuters import embed_episode_from_embeds

    num_tokens = input_ids.shape[1]
    token_ids = input_ids[0].tolist()
    content_positions = list(range(1, num_tokens - 1))

    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    bos_emb = word_embeddings(torch.tensor(bos_id, device=device))
    eos_emb = word_embeddings(torch.tensor(eos_id, device=device))
    pad_emb = word_embeddings(torch.tensor(pad_id, device=device))

    content_ids_t = torch.tensor([token_ids[p] for p in content_positions], device=device)
    doc_mean_vec = word_embeddings(content_ids_t).mean(dim=0)

    if config.MODE != "chunk32":
        all_embeds = []
        all_mask = []
        for keep in keep_lists:
            keep_set = set(keep)
            seq = [
                word_embeddings(torch.tensor(token_ids[p], device=device)) if p in keep_set else doc_mean_vec
                for p in range(num_tokens)
            ]
            all_embeds.append(torch.stack(seq))
            all_mask.append(attention_mask[0].to(device))

        embeddings = []
        for i in range(0, len(all_embeds), BATCH_SIZE):
            batch_embeds = torch.stack(all_embeds[i:i+BATCH_SIZE]).unsqueeze(1)  # (B, E=1, L, H)
            batch_mask = torch.stack(all_mask[i:i+BATCH_SIZE]).unsqueeze(1)  # (B, E=1, L)
            embeddings.append(embed_episode_from_embeds(batch_embeds, batch_mask))
        return torch.cat(embeddings, dim=0)

    piece_ranges = _piece_ranges_for(num_tokens)

    all_embeds = []
    all_mask = []
    for keep in keep_lists:
        keep_set = set(keep)
        piece_embeds = []
        piece_masks = []
        for piece_range in piece_ranges:
            seq = [bos_emb]
            for p in piece_range:
                if p in keep_set:
                    seq.append(word_embeddings(torch.tensor(token_ids[p], device=device)))
                else:
                    seq.append(doc_mean_vec)
            seq.append(eos_emb)
            mask = [1] * len(seq)
            pad_len = 32 - len(seq)
            seq = seq + [pad_emb] * pad_len
            mask = mask + [0] * pad_len
            piece_embeds.append(torch.stack(seq))
            piece_masks.append(torch.tensor(mask, device=device))
        all_embeds.append(torch.stack(piece_embeds))
        all_mask.append(torch.stack(piece_masks))

    embeddings = []
    for i in range(0, len(all_embeds), BATCH_SIZE):
        batch_embeds = torch.stack(all_embeds[i:i+BATCH_SIZE])
        batch_mask = torch.stack(all_mask[i:i+BATCH_SIZE])
        embeddings.append(embed_episode_from_embeds(batch_embeds, batch_mask))
    return torch.cat(embeddings, dim=0)

def embed_from_ids(input_ids, attention_mask):
    # input_ids, attention_mask: shape (batch, seq_len), each row a separate document (episode length 1)
    # processed in batches of BATCH_SIZE to bound memory use
    embeddings = []
    for i in range(0, input_ids.shape[0], BATCH_SIZE):
        chunk_ids = input_ids[i:i+BATCH_SIZE].reshape(-1, 1, input_ids.shape[1]).to(device)
        chunk_mask = attention_mask[i:i+BATCH_SIZE].reshape(-1, 1, attention_mask.shape[1]).to(device)
        with torch.no_grad():
            embeddings.append(model(input_ids=chunk_ids, attention_mask=chunk_mask).cpu())
    return torch.cat(embeddings, dim=0)
