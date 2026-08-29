# loads LUAR model into ~/.cache/huggingface/hub

from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model = AutoModel.from_pretrained("rrivera1849/LUAR-MUD", trust_remote_code=True)
model.eval()

def embed(text, max_length=32):
    tokenized = tokenizer(
        [text],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    tokenized["input_ids"] = tokenized["input_ids"].reshape(1, 1, -1)
    tokenized["attention_mask"] = tokenized["attention_mask"].reshape(1, 1, -1)
    with torch.no_grad():
        return model(**tokenized).squeeze(0)