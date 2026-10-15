# training batches of episodes: each episode is pieces_per_episode 32-token pieces of one
# author's training texts, cut exactly like model.embed_chunked cuts them

import torch
from model import chunk_pieces

AUTHORS_PER_BATCH = {"reuters": 16, "darkreddit": 10}
PIECES_PER_EPISODE = 16

# text -> (pieces ids, pieces masks), so every training text is tokenized only once
_pieces_cache = {}


def text_pieces(text):
    if text not in _pieces_cache:
        _pieces_cache[text] = chunk_pieces(text)
    return _pieces_cache[text]


def sample_episode(texts, pieces_per_episode, rng):
    # pieces drawn uniformly from all pieces of the given texts, without replacement unless
    # there are fewer pieces than needed
    pool = [(ids, mask) for text in texts for ids, mask in zip(*text_pieces(text))]
    if len(pool) >= pieces_per_episode:
        chosen = rng.sample(pool, pieces_per_episode)
    else:
        chosen = [rng.choice(pool) for _ in range(pieces_per_episode)]
    return [ids for ids, _ in chosen], [mask for _, mask in chosen]


def sample_batch(split, authors_per_batch, pieces_per_episode, rng):
    # split: author -> {"train": [texts], ...} as returned by splits.get_split
    # rng: random.Random
    # returns input_ids, attention_mask of shape (2 * authors_per_batch, pieces_per_episode, 32)
    # and labels of shape (2 * authors_per_batch,), the author's index in sorted(split);
    # rows 2k and 2k+1 are the two episodes of the k-th sampled author
    authors = sorted(split)
    batch_authors = rng.sample(range(len(authors)), authors_per_batch)

    all_ids = []
    all_mask = []
    labels = []
    for author_index in batch_authors:
        texts = [t for t in split[authors[author_index]]["train"] if text_pieces(t)[0]]
        # the two episodes get disjoint halves of the author's texts, so they share no text;
        # an author with a single usable text has to use it for both
        shuffled = rng.sample(texts, len(texts))
        if len(shuffled) >= 2:
            halves = [shuffled[:len(shuffled) // 2], shuffled[len(shuffled) // 2:]]
        else:
            halves = [shuffled, shuffled]
        for half in halves:
            ids, mask = sample_episode(half, pieces_per_episode, rng)
            all_ids.append(ids)
            all_mask.append(mask)
            labels.append(author_index)

    return torch.tensor(all_ids), torch.tensor(all_mask), torch.tensor(labels)
