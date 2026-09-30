DATASET = "reuters"
MODEL = "luar-mud"
USE_CHUNKS = True
RESULTS_DIR = f"results/{DATASET}_{MODEL}" + ("_chunk32" if USE_CHUNKS else "")
CACHE_DIR = f"cache/{DATASET}_{MODEL}"
