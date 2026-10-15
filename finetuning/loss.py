import torch
import torch.nn.functional as F

DEFAULT_TEMPERATURE = 0.07


def supcon_loss(embeddings, labels, temperature=DEFAULT_TEMPERATURE):
    # supervised contrastive loss (Khosla et al. 2020, L_out):
    # embeddings: (N, D) episode embeddings, labels: (N,) author index per episode
    # for each anchor, all other episodes with the same label are positives, the rest negatives
    z = F.normalize(embeddings, dim=1)
    logits = z @ z.T / temperature

    n = z.shape[0]
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    positive_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~self_mask

    # the anchor itself is excluded from the softmax denominator
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    # anchors without any positive contribute nothing
    num_positives = positive_mask.sum(dim=1)
    has_positive = num_positives > 0
    mean_log_prob_pos = log_prob.masked_fill(~positive_mask, 0).sum(dim=1)[has_positive] / num_positives[has_positive]
    return -mean_log_prob_pos.mean()
