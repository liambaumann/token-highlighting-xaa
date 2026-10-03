DATASET = "reuters"
MODEL = "luar-mud"

# "plain": one 512-token sequence per document
# "chunk32": content split into 30-token pieces (+<s></s>, padded to 32), one episode
# "single32": one random 30-token content excerpt (+<s></s>) per document
MODE = "chunk32"

SINGLE32_SEEDS = [0, 1, 2]

MODEL_MODE_NAME = f"{MODEL}_{MODE}"

# run.py (occlusion/masking) and retrieval.py (author retrieval) are different tests over
# the same dataset/model/mode, so they get separate result trees; dataset and model+mode
# each get their own folder level, so e.g. "reuters" and "chunk32" aren't welded together
MASKING_RESULTS_DIR = f"results/masking/{DATASET}/{MODEL_MODE_NAME}"
RETRIEVAL_RESULTS_DIR = f"results/retrieval/{DATASET}/{MODEL_MODE_NAME}"

CACHE_DIR = f"cache/{DATASET}_{MODEL}"
