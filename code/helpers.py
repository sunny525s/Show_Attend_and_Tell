import torch
import numpy as np
from pathlib import Path


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

def save_checkpoint(
    epoch: int,
    epochs_since_improvement: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    encoder_optimizer: torch.optim.Optimizer | None,
    decoder_optimizer: torch.optim.Optimizer,
    bleu4: float,
    is_best: bool,
    ckpt_dir: str = "checkpoints",
    prefix: str = "image_captioning"
):
    """
    Save model + optimizer state_dicts.

    Args:
        epoch: current epoch (1-indexed).
        epochs_since_improvement: epochs since last BLEU-4 improvement.
        encoder: the encoder network.
        decoder: the decoder network.
        encoder_optimizer: optimizer for encoder (or None).
        decoder_optimizer: optimizer for decoder.
        bleu4: current BLEU-4 score.
        is_best: True if this is the best model so far.
        ckpt_dir: directory to save checkpoints into.
        prefix: filename prefix (no extension).
    """
    ckpt_path = Path(ckpt_dir)
    ckpt_path.mkdir(exist_ok=True)

    state = {
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_improvement,
        "bleu-4": bleu4,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "decoder_optimizer_state_dict": decoder_optimizer.state_dict(),
    }
    # Only include encoder optimizer if it exists
    if encoder_optimizer is not None:
        state["encoder_optimizer_state_dict"] = encoder_optimizer.state_dict()

    # Filename with zero-padded epoch number
    fname = f"{prefix}_epoch{epoch:02d}.pth"
    full_path = ckpt_path / fname
    torch.save(state, full_path)
    print(f"Checkpoint saved: {full_path}")

    # If this is the best model so far, overwrite the "best" symlink
    if is_best:
        best_path = ckpt_path / f"{prefix}_best.pth"
        torch.save(state, best_path)
        print(f"*** New best model saved: {best_path} ***")
