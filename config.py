# "reuters": Reuters C50
# "darkreddit": DarkReddit
DATASET = "darkreddit"

# like "luar-mud", "luar-mud-ft-darkreddit"
#MODEL = "luar-mud-ft-darkreddit"
MODEL = "luar-mud-ft-darkreddit-top4"

# "plain": one 512-token sequence per document
# "chunk32": content split into 30-token pieces (+<s></s>, padded to 32), one episode
# "single32": one random 30-token content excerpt (+<s></s>) per document
MODE = "plain"

# how dropped (occluded) positions are filled in masking's embed_variants:
# "delete": position removed (today's behaviour)
# "mask": position kept, token id replaced with tokenizer.mask_token_id
# "doc_mean": position kept, word embedding replaced with the mean word embedding of the
#             document's own (unperturbed) content tokens
PERTURBATION = "doc_mean"

SINGLE32_SEEDS = [0, 1, 2]

MODEL_MODE_NAME = f"{MODEL}_{MODE}"

# run.py (occlusion/masking) and retrieval.py (author retrieval) are different tests over
# the same dataset/model/mode, so they get separate result trees; dataset and model+mode
# each get their own folder level, so e.g. "reuters" and "chunk32" aren't welded together.
# masking results also fork on PERTURBATION; retrieval doesn't perturb anything, so its
# paths stay as they were
MASKING_RESULTS_DIR = f"results/masking/{DATASET}/{PERTURBATION}/{MODEL_MODE_NAME}"
RETRIEVAL_RESULTS_DIR = f"results/retrieval/{DATASET}/{MODEL_MODE_NAME}"

CACHE_DIR = f"cache/{DATASET}_{MODEL}"
