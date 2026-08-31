import os
import torch
from luar_model import embed
from reuters_data import load_author_texts, get_centroids


def get_similarity_matrix(authors, centroids):
    sims = []
    true_labels = []
    for i, author in enumerate(authors):
        print("testing", author)
        query_texts = load_author_texts(f"data/C50test/{author}")
        for query_text in query_texts:
            query_embedding = embed(query_text)
            row = [
                torch.nn.functional.cosine_similarity(query_embedding, centroids[a], dim=0).item()
                for a in authors
            ]
            sims.append(row)
            true_labels.append(i)

    sims = torch.tensor(sims)
    true_labels = torch.tensor(true_labels)
    return sims, true_labels


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


if __name__ == "__main__":
    authors = sorted(os.listdir("data/C50train"))[:15] # limit authors here, e.g. [:20]
    centroids = get_centroids(authors)
    sims, true_labels = get_similarity_matrix(authors, centroids)

    correct, total = top1_accuracy(sims, true_labels)
    correct_top5, total_top5 = soft_top_x_accuracy(sims, true_labels, 5)
    correct_top10, total_top10 = soft_top_x_accuracy(sims, true_labels, 10)
    mAP = mean_avg_precision(sims, true_labels)
    print(f"\ntop-1 accuracy: {correct}/{total} (~{correct/total:.2%})")
    print(f"soft top-5 accuracy: {correct_top5}/{total_top5} (~{correct_top5/total_top5:.2%})")
    print(f"soft top-10 accuracy: {correct_top10}/{total_top10} (~{correct_top10/total_top10:.2%})")
    print(f"mean average precision: {mAP:.2%}")