import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import nltk
from nltk.translate.meteor_score import meteor_score
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from helpers import accuracy, clip_gradient, compute_meteor_scores
from decoder import Decoder
from encoder import ImageEncoder

# Download nltk
nltk.download('wordnet')

# Global variables
embed_dim = 512
attention_dim = 512
decoder_dim = 512
encoder_dim = 1920
encoder_lr = 1e-4
decoder_lr = 4e-4
grad_clip = 5.
alpha_c = 1.
num_epochs = 20
fine_tune_encoder = False
lr_decay_factor = 0.8
lr_decay_patience = 8
vocab_size = 0
word2id = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup
cudnn.benchmark = True
scaler = GradScaler()

def set_hyperparameter(train_config):
    global embed_dim, attention_dim, decoder_dim, encoder_dim
    global encoder_lr, decoder_lr, grad_clip, alpha_c
    global num_epochs, fine_tune_encoder, lr_decay_factor, lr_decay_patience
    global word2id, vocab_size, device

    embed_dim = train_config['embed_dim']
    attention_dim = train_config['attention_dim']
    decoder_dim = train_config['decoder_dim']
    encoder_dim = train_config['encoder_dim']
    encoder_lr = train_config['encoder_lr']
    decoder_lr = train_config['decoder_lr']
    grad_clip = train_config['grad_clip']
    alpha_c = train_config['alpha_c']
    num_epochs = train_config['num_epochs']
    fine_tune_encoder = train_config['fine_tune_encoder']
    lr_decay_factor = train_config['lr_decay_factor']
    lr_decay_patience = train_config['lr_decay_patience']
    word2id = train_config['word2id']
    vocab_size = len(word2id)
    device = train_config['device']

def train_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer):
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

        with autocast():
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, cap_lens)
            targets = caps_sorted[:, 1:]

            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores_packed, targets_packed) + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(decoder_optimizer)
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                scaler.unscale_(encoder_optimizer)
                clip_gradient(encoder_optimizer, grad_clip)

        scaler.step(decoder_optimizer)
        if encoder_optimizer is not None:
            scaler.step(encoder_optimizer)

        scaler.update()

        top5 = accuracy(scores_packed, targets_packed, 5)
        losses.append(loss.item())
        top5accs.append(top5)

    return np.mean(losses), np.mean(top5accs)

def val_epoch(val_loader, encoder, decoder, criterion):
    losses = []
    top5accs = []
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    references = []
    hypotheses = []

    with torch.no_grad():
        for imgs, caps, cap_lens, all_caps in tqdm(val_loader, total=len(val_loader)):
            imgs = imgs.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            cap_lens = cap_lens.to(device, non_blocking=True)

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, cap_lens)
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = criterion(scores, targets) + alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
            top5 = accuracy(scores, targets, 5)

            losses.append(loss.item())
            top5accs.append(top5)

            sort_ind = sort_ind.cpu()
            all_caps = all_caps[sort_ind]
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                img_captions = list(map(
                    lambda caption: [word for word in caption if word not in {word2id['<start>'], word2id['<pad>']}],
                    img_caps))
                references.append(img_captions)

            _, preds = torch.max(scores_copy, dim=2)
            preds = preds.tolist()
            temp_preds = [preds[j][:decode_lengths[j]] for j in range(len(preds))]
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

def main_train_loop(train_loader, val_loader):
    decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim)
    decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)

    encoder = ImageEncoder()
    if fine_tune_encoder:
        encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        encoder_optimizer, mode='max', factor=lr_decay_factor, patience=lr_decay_patience
    ) if fine_tune_encoder else None
    decoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        decoder_optimizer, mode='max', factor=lr_decay_factor, patience=lr_decay_patience
    )

    criterion = nn.CrossEntropyLoss().to(device)

    best_bleus = np.zeros(4)
    best_avg = 0.
    best_meteor = 0.
    epochs_since_improvement = 0

    for epoch in range(1, num_epochs + 1):
        loss_train, acc_train = train_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
        loss_val, acc_val, bleu_vals, m_score = val_epoch(val_loader, encoder, decoder, criterion)

        decoder_lr_scheduler.step(bleu_vals[3])
        if fine_tune_encoder:
            encoder_lr_scheduler.step(bleu_vals[3])

        score_avg = (np.sum(bleu_vals) + m_score) / 5
        is_best = score_avg > best_avg

        best_bleus = np.maximum(bleu_vals, best_bleus)
        best_meteor = max(m_score, best_meteor)
        best_avg = max(score_avg, best_avg)

        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        print('-' * 50)
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {loss_train:.4f}, Train Acc: {acc_train:.2f}%')
        print(f'Valid Loss: {loss_val:.4f}, Valid Acc: {acc_val:.2f}%')
        print(f'Best BLEUs: 1: {best_bleus[3]:.4f}, 2: {best_bleus[2]:.4f}, 3: {best_bleus[1]:.4f}, 4: {best_bleus[0]:.4f}')
        print(f'Best METEOR: {best_meteor:.4f}')
        print('-' * 50)
