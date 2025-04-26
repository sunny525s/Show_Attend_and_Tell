# model parameters
embed_dim = 512      # dim of word embeddings
attention_dim = 512  # dim of attention linear layers
decoder_dim = 512    # dim of decoder RNN
encoder_dim = 1920   # 512 for VGG, 2048 for ResNet, 1920 for DenseNet, 1024 for GoogLeNet, 3 for AlexNet
encoder_lr = 1e-4    # learning rate for encoder if fine-tuning
decoder_lr = 4e-4    # learning rate for decoder
grad_clip = 5.       # clip gradients at an absolute value of
alpha_c = 1.         # regularization parameter for 'doubly stochastic attention'
vocab_size = len(word2id)

lr_decay_factor = 0.8
lr_decay_patience = 8

num_epochs = 20
epochs_since_improvement = 0

fine_tune_encoder = False  # fine-tune encoder?
cudnn.benchmark = True     # set to true only if inputs to model are fixed size

# model setup
decoder = Decoder(attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim)
decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)

encoder = ImageEncoder()
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
    # save the checkpoint
    save_checkpoint(
        epoch=epoch,
        epochs_since_improvement=epochs_since_improvement,
        encoder=encoder,
        decoder=decoder,
        encoder_optimizer=encoder_optimizer if fine_tune_encoder else None,
        decoder_optimizer=decoder_optimizer,
        bleu4=bleu_vals[0],    # BLEU-4 is the first element
        is_best=is_best
    )
