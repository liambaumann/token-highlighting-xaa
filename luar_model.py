# loads LUAR model into models/ next to this file (downloaded on first run)

import os

# all Hugging Face files (model, custom code, caches) go to models/ instead of ~/.cache/huggingface
# must be set before transformers is imported
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

from transformers import AutoTokenizer, AutoModel
import torch

# use GPU if available (on VSC)
device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
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
    
def embed_from_ids(input_ids, attention_mask):
    # input_ids, attention_mask: shape (batch, seq_len), each row a separate document (episode length 1)
    # processed in chunks of BATCH_SIZE to bound memory use
    embeddings = []
    for i in range(0, input_ids.shape[0], BATCH_SIZE):
        chunk_ids = input_ids[i:i+BATCH_SIZE].reshape(-1, 1, input_ids.shape[1]).to(device)
        chunk_mask = attention_mask[i:i+BATCH_SIZE].reshape(-1, 1, attention_mask.shape[1]).to(device)
        with torch.no_grad():
            embeddings.append(model(input_ids=chunk_ids, attention_mask=chunk_mask).cpu())
    return torch.cat(embeddings, dim=0)
