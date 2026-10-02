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

tokenizer = AutoTokenizer.from_pretrained(MODELS[config.MODEL], trust_remote_code=True)
model = AutoModel.from_pretrained(MODELS[config.MODEL], trust_remote_code=True)
model.eval()
model.to(device)

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

def embed_chunked(texts):
    # tokenize each text into 30-token pieces (plus <s>/</s>, padded to 32), all pieces from
    # all texts form one episode, so the model aggregates across pieces the same way it
    # aggregates across texts in embed_episode
    all_pieces_ids = []
    all_pieces_mask = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"][:510]
        for i in range(0, len(ids), 30):
            piece = [tokenizer.bos_token_id] + ids[i:i+30] + [tokenizer.eos_token_id]
            mask = [1] * len(piece)
            pad_len = 32 - len(piece)
            piece = piece + [tokenizer.pad_token_id] * pad_len
            mask = mask + [0] * pad_len
            all_pieces_ids.append(piece)
            all_pieces_mask.append(mask)

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

def embed_variants(input_ids, attention_mask, keep_lists):
    # input_ids, attention_mask: shape (1, num_tokens), a single tokenized document (<s>, content, </s>)
    # keep_lists: list of variants, each a sorted list of positions (into input_ids) to keep
    if config.MODE != "chunk32":
        variant_ids = torch.cat([input_ids[:, keep] for keep in keep_lists], dim=0)
        variant_mask = torch.cat([attention_mask[:, keep] for keep in keep_lists], dim=0)
        return embed_from_ids(variant_ids, variant_mask)

    num_tokens = input_ids.shape[1]
    content_end = num_tokens - 2  # last content position (num_tokens - 1 is </s>)
    num_pieces = max(1, -(-content_end // 30))  # ceil(content_end / 30)
    piece_ranges = [
        range(1 + 30 * j, min(30 + 30 * j, content_end) + 1)
        for j in range(num_pieces)
    ]

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
