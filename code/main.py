from data_prep import generate_json_data
from dataset import CaptionDataset, create_dataloaders
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import json
import torch.backends.cudnn as cudnn
from decoder import Decoder
import torch.optim as optim
from densenet_encoder import ImageEncoder
from torch.amp import GradScaler
import torch
import torch.nn as nn
import numpy as np
from training import train_epoch, val_epoch
from helpers import save_checkpoint, get_wordmap
import nltk
from config import *


def setup_models(vocab_size, device, fine_tune_encoder=False, checkpoint_path=CHECKPOINT_PATH):
    encoder = ImageEncoder()
    encoder.fine_tune(fine_tune_encoder)

    decoder = Decoder(ATTENTION_DIM, EMBED_DIM, DECODER_DIM, vocab_size, ENCODER_DIM)

    encoder_optimizer = (
        optim.Adam(
            params=filter(lambda p: p.requires_grad, encoder.parameters()),
            lr=ENCODER_LR,
        )
        if fine_tune_encoder
        else None
    )
    decoder_optimizer = optim.Adam(
        params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=DECODER_LR
    )

    # Load checkpoint if existing
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print(f"Loading checkpoint from {checkpoint_path} at epoch  {checkpoint['epoch']}")
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        decoder_optimizer.load_state_dict(checkpoint    ['decoder_optimizer_state_dict'])
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        if fine_tune_encoder:
            encoder_optimizer.load_state_dict(checkpoint    ['encoder_optimizer_state_dict'])
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting  from scratch.")

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    return encoder, decoder, encoder_optimizer, decoder_optimizer


def setup_schedulers(encoder_optimizer, decoder_optimizer, fine_tune_encoder=False):
    encoder_lr_scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            encoder_optimizer,
            mode="max",
            factor=LR_DECAY_FACTOR,
            patience=LR_DECAY_PATIENCE,
        )
        if fine_tune_encoder
        else None
    )

    decoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer,
        mode="max",
        factor=LR_DECAY_FACTOR,
        patience=LR_DECAY_PATIENCE,
    )
    return encoder_lr_scheduler, decoder_lr_scheduler


def run_training(
    train_loader,
    val_loader,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    encoder_lr_scheduler,
    decoder_lr_scheduler,
    device,
    num_epochs=NUM_EPOCHS,
    patience=PATIENCE
):
    criterion = nn.CrossEntropyLoss().to(device)
    best_bleus = np.zeros(4)
    best_avg = 0.0
    best_meteor = 0.0
    epochs_since_improvement = 0

    scaler = GradScaler(device.type)
    for epoch in range(1, num_epochs + 1):
        loss_train, acc_train = train_epoch(
            train_loader,
            encoder,
            decoder,
            criterion,
            encoder_optimizer,
            decoder_optimizer,
            device,
            scaler,
        )
        loss_val, acc_val, bleu_vals, m_score = val_epoch(
            val_loader, encoder, decoder, criterion, device
        )

        decoder_lr_scheduler.step(bleu_vals[3])
        if encoder_lr_scheduler:
            encoder_lr_scheduler.step(bleu_vals[3])

        score_avg = (np.sum(bleu_vals) + m_score) / 5
        is_best = score_avg > best_avg
        best_bleus = np.maximum(bleu_vals, best_bleus)
        best_meteor = max(m_score, best_meteor)
        best_avg = max(score_avg, best_avg)
        epochs_since_improvement = 0 if is_best else epochs_since_improvement + 1

        print("-" * 40)
        print(
            f"epoch: {epoch}, train loss: {loss_train:.4f}, train acc: {acc_train:.2f}%, valid loss: {loss_val:.4f}, valid acc: {acc_val:.2f}%, best BLEU-1: {best_bleus[0]:.4f}, best BLEU-2: {best_bleus[1]:.4f}, best BLEU-3: {best_bleus[2]:.4f}, best BLEU-4: {best_bleus[3]:.4f}, best METEOR: {best_meteor:.4f}"
        )
        print("-" * 40)

        save_checkpoint(
            epoch=epoch,
            epochs_since_improvement=epochs_since_improvement,
            encoder=encoder,
            decoder=decoder,
            encoder_optimizer=encoder_optimizer,
            decoder_optimizer=decoder_optimizer,
            bleu_scores=bleu_vals,
            meteor_score=m_score,
            val_acc = acc_val,
            train_acc = acc_train,
            is_best=is_best,
        )

        if epochs_since_improvement >= patience:
            print(f"Early stopping: No BLEU-4 improvement for {patience} epochs.")
            break


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    nltk.download("wordnet", quiet=True)

    # TODO: generate_json_data

    transform = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )
    train_loader, val_loader, test_loader = create_dataloaders(transform, CAPTION_PATH)

    word2id = get_wordmap()
    vocab_size = len(word2id)

    encoder, decoder, encoder_optimizer, decoder_optimizer = setup_models(
        vocab_size, device, fine_tune_encoder=FINE_TUNE_ENCODER
    )
    encoder_lr_scheduler, decoder_lr_scheduler = setup_schedulers(
        encoder_optimizer, decoder_optimizer, fine_tune_encoder=FINE_TUNE_ENCODER
    )

    run_training(
        train_loader,
        val_loader,
        encoder,
        decoder,
        encoder_optimizer,
        decoder_optimizer,
        encoder_lr_scheduler,
        decoder_lr_scheduler,
        device,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE
    )
