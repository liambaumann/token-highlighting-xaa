# one-off: Reuters as a 10-way task, fixed author subset
import random
import config
from model import embed_chunked
from data import load_authors, get_centroids
from retrieval import get_similarity_matrix, print_retrieval_metrics

assert config.DATASET == "reuters" and config.MODE in ("plain", "chunk32")

authors = sorted(random.Random(0).sample(load_authors("reuters"), 10))
print("authors:", authors)

if config.MODE == "chunk32":
    centroids = get_centroids("reuters", load_authors("reuters"), embed_fn=embed_chunked, cache_name="centroids_chunk32.pt")
    sims, labels = get_similarity_matrix(authors, centroids, embed_fn=lambda t: embed_chunked([t]))
else:
    centroids = get_centroids("reuters", load_authors("reuters"))
    sims, labels = get_similarity_matrix(authors, centroids)

print_retrieval_metrics(sims, labels)