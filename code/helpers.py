import torch
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def compute_meteor_scores(references_ids, hypotheses_ids, id2word):
    meteor_scores = []
    # printed = False
    for refs, hyp in zip(references_ids, hypotheses_ids):
        # Convert refs and hyp to strings
        refs_str = [[id2word[tok] for tok in ref] for ref in refs]
        hyp_str = [id2word[tok] for tok in hyp]

        # if not printed:
        #     print(refs_str)
        #     print(hyp_str)
        #     printed = True

        score = meteor_score(refs_str, hyp_str)
        meteor_scores.append(score)

    return sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0