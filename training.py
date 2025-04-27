import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.amp import autocast
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from helpers import accuracy, clip_gradient, compute_meteor_scores, get_wordmap
from decoder import Decoder
from densenet_encoder import ImageEncoder
import os
import json
from config import *


def train_epoch(
    train_loader,
    encoder,
    decoder,
    criterion,
    encoder_optimizer,
    decoder_optimizer,
    device,
    scaler,
):

    losses = []
    top5accs = []
    decoder.train()
    encoder.train()

    for imgs, caps, cap_lens in tqdm(train_loader, total=len(train_loader)):
        imgs = imgs.to(device, non_blocking=True)
        caps = caps.to(device, non_blocking=True)
        cap_lens = cap_lens.to(device, non_blocking=True)

        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()

        with autocast("cuda"):
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, cap_lens
            )
            targets = caps_sorted[:, 1:]

            scores_packed = pack_padded_sequence(
                scores, decode_lengths, batch_first=True
            ).data
            targets_packed = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = (
                criterion(scores_packed, targets_packed)
                + ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            )

        scaler.scale(loss).backward()

        if GRAD_CLIP is not None:
            scaler.unscale_(decoder_optimizer)
            clip_gradient(decoder_optimizer, GRAD_CLIP)
            if encoder_optimizer is not None:
                scaler.unscale_(encoder_optimizer)
                clip_gradient(encoder_optimizer, GRAD_CLIP)

        scaler.step(decoder_optimizer)
        if encoder_optimizer is not None:
            scaler.step(encoder_optimizer)

        scaler.update()

        top5 = accuracy(scores_packed, targets_packed, 5)
        losses.append(loss.item())
        top5accs.append(top5)

    return np.mean(losses), np.mean(top5accs)


def val_epoch(val_loader, encoder, decoder, criterion, device):
    losses = []
    top5accs = []
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    references = []
    hypotheses = []

    word2id = get_wordmap()

    with torch.no_grad():
        for imgs, caps, cap_lens, all_caps in tqdm(val_loader, total=len(val_loader)):
            imgs = imgs.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            cap_lens = cap_lens.to(device, non_blocking=True)

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(
                imgs, caps, cap_lens
            )
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(
                targets, decode_lengths, batch_first=True
            ).data

            loss = (
                criterion(scores, targets)
                + ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            )
            top5 = accuracy(scores, targets, 5)

            losses.append(loss.item())
            top5accs.append(top5)

            sort_ind = sort_ind.cpu()
            all_caps = all_caps[sort_ind]
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(
                    map(
                        lambda caption: [
                            word
                            for word in caption
                            if word not in {word2id["<start>"], word2id["<pad>"]}
                        ],
                        img_caps,
                    )
                )
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = [preds[j][: decode_lengths[j]] for j in range(len(preds))]
            hypotheses.extend(temp_preds)

        assert len(references) == len(hypotheses)
        bleu_scores = [
            corpus_bleu(references, hypotheses),
            corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)),
            corpus_bleu(references, hypotheses, weights=(0.25, 0.5, 0, 0)),
            corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0)),
        ]
        m_score = compute_meteor_scores(references, hypotheses)

    return np.mean(losses), np.mean(top5accs), bleu_scores, m_score
