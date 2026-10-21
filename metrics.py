# author retrieval metrics over a similarity matrix (one row per query, one column per author)


def top1_accuracy(sims, true_labels):
    predictions = sims.argmax(dim=1)
    correct = (predictions == true_labels).sum().item()
    total = len(true_labels)
    return correct, total


def soft_top_x_accuracy(sims, true_labels, x):
    top_x_predictions = sims.topk(x, dim=1).indices
    correct = sum(true_labels[i] in top_x_predictions[i] for i in range(len(true_labels)))
    total = len(true_labels)
    return correct, total


def mean_avg_precision(sims, true_labels):
    reciprocal_ranks = []
    for i in range(len(true_labels)):
        row = sims[i]
        true_col = true_labels[i].item()
        sorted_indices = row.argsort(descending=True)
        rank = (sorted_indices == true_col).nonzero(as_tuple=True)[0].item() + 1
        reciprocal_ranks.append(1.0 / rank)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)
