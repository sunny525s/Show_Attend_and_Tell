import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

from helpers import accuracy, clip_gradient, compute_meteor_scores

from decoder import Decoder
from encoder import ImageEncoder

scaler = GradScaler()

# model parameters
embed_dim = 512      # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512    # dimension of decoder RNN
encoder_dim = 2048
encoder_lr = 1e-4    # learning rate for encoder if fine-tuning
decoder_lr = 4e-4    # learning rate for decoder
grad_clip = 5.       # clip gradients at an absolute value of
alpha_c = 1.         # regularization parameter for 'doubly stochastic attention'

lr_decay_factor = 0.8
lr_decay_patience = 8

num_epochs = 10
epochs_since_improvement = 0

vocab_size = 0

fine_tune_encoder = False  # fine-tune encoder?
cudnn.benchmark = True     # set to true only if inputs to model are fixed size

word2id = {}
device = None

def train_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer):
    losses = []
    top5accs = []

    decoder.train()
    encoder.train()

    for i, (imgs, caps, cap_lens) in enumerate(tqdm(train_loader, total=len(train_loader))):
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

            scores_packed = pack_padded_sequence(scores, decode_lengths.cpu(), batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True).data

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
        for i, (imgs, caps, cap_lens, all_caps) in enumerate(tqdm(val_loader, total=len(val_loader))):
            imgs = imgs.to(device, non_blocking=True)
            caps = caps.to(device, non_blocking=True)
            cap_lens = cap_lens.to(device, non_blocking=True)

            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, cap_lens)
            targets = caps_sorted[:, 1:]

            scores_copy = scores.clone()

            scores = pack_padded_sequence(scores, decode_lengths.cpu(), batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths.cpu(), batch_first=True).data

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
        bleu_scores = [corpus_bleu(references, hypotheses), 
                       corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0)),
                       corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0)),
                       corpus_bleu(references, hypotheses, weights=(1.0, 0, 0, 0))]
        # print("references: ", references)
        # print("hypotheses: ", hypotheses)
        m_score = compute_meteor_scores(references, hypotheses)

    return np.mean(losses), np.mean(top5accs), bleu_scores, m_score

def set_hyperparameter(train_config):
    # model setup
    global embed_dim 
    embed_dim = train_config['embed_dim']
    global attention_dim
    attention_dim = train_config['attention_dim']
    global decoder_dim
    decoder_dim = train_config['decoder_dim']
    global encoder_dim
    encoder_dim = train_config['encoder_dim']
    global encoder_lr
    encoder_lr = train_config['encoder_lr']
    global decoder_lr
    decoder_lr = train_config['decoder_lr']
    global grad_clip
    grad_clip = train_config['grad_clip']
    global alpha_c 
    alpha_c = train_config['alpha_c']
    global num_epochs 
    num_epochs = train_config['num_epochs']
    global fine_tune_encoder 
    fine_tune_encoder = train_config['fine_tune_encoder']
    global lr_decay_factor 
    lr_decay_factor = train_config['lr_decay_factor']
    global lr_decay_patience
    lr_decay_patience = train_config['lr_decay_patience']
    global word2id
    word2id = train_config['word2id']
    global vocab_size
    vocab_size = len(word2id)
    global device
    device = train_config['device']

def main_train_loop(train_loader, val_loader):
    epochs_since_improvement = 0

    decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim)
    decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)

    encoder = ImageEncoder()
    if fine_tune_encoder:
        encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()), lr=encoder_lr) if fine_tune_encoder else None

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # lr scheduler
    encoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='max', factor=lr_decay_factor, patience=lr_decay_patience) if fine_tune_encoder else None
    decoder_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode='max', factor=lr_decay_factor, patience=lr_decay_patience)

    # criterion for loss
    criterion = nn.CrossEntropyLoss().to(device)

    # loop
    best_bleus = np.zeros(4)
    best_avg = 0.
    best_meteor = 0.
    for epoch in range(1, num_epochs + 1):
        loss_train, acc_train = train_epoch(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer)
        loss_val, acc_val, bleu_vals, m_score = val_epoch(val_loader, encoder, decoder, criterion)

        # reduce the learning rate on plateau
        decoder_lr_scheduler.step(bleu_vals[3])
        if fine_tune_encoder:
            encoder_lr_scheduler.step(bleu_vals[3])

        # check if there was an improvement
        score_avg = (np.sum(bleu_vals) + m_score) / 5
        is_best = score_avg > best_avg
        best_bleus = np.maximum(bleu_vals, best_bleus)
        best_meteor = max(m_score, best_meteor)
        best_avg = max(score_avg, best_avg)
        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        print('-' * 40)
        print(f'epoch: {epoch}, train loss: {loss_train:.4f}, train acc: {acc_train:.2f}%, valid loss: {loss_val:.4f}, valid acc: {acc_val:.2f}%, best BLEU-1: {best_bleus[3]:.4f}, best BLEU-2: {best_bleus[2]:.4f}, best BLEU-3: {best_bleus[1]:.4f}, best BLEU-4: {best_bleus[0]:.4f}, best METEOR: {best_meteor:.4f}')
        print('-' * 40)