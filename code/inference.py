import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import skimage.transform
import matplotlib.pyplot as plt
from torchvision import transforms

def _load_and_preprocess(image_path: str, image_size: int = 256):
    """Load image from disk, resize, normalize and return a tensor."""
    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (image_size, image_size))
    assert img.shape == (image_size, image_size, 3)
    assert np.max(img) <= 255

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(img)  # (3, H, W)

def _plot_attention(image_path: str, caption: list, alphas: np.ndarray, grid_size: int, upsample: int = 24):
    """Display attention maps over the image for each word in caption."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize([grid_size * upsample]*2, Image.LANCZOS)

    num_words = len(caption)
    cols = 5
    rows = int(np.ceil(num_words / cols))
    plt.figure(figsize=(cols*4, rows*3))

    for t, word in enumerate(caption):
        plt.subplot(rows, cols, t+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(word, fontsize=12)

        alpha_map = alphas[t]
        alpha_up = skimage.transform.pyramid_expand(
            alpha_map, upscale=upsample, sigma=8
        )
        plt.imshow(alpha_up, alpha=0.8, cmap='gray')
    plt.tight_layout()
    plt.show()


def generate_caption(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    image_path: str,
    word_map: dict,
    beam_size: int = 5,
    max_caption_len: int = 50,
    return_attention: bool = True,
    device: torch.device = None,
):
    """
    Generate an image caption with beam search and optional attention visualization.

    Args:
        encoder, decoder       : trained models
        image_path (str)       : path to the input image
        word_map (dict)        : token->index mapping
        beam_size (int)        : how many beams to keep
        max_caption_len (int)  : early stop if we reach length
        return_attention (bool): plot attention
        device (torch.device)  : cpu/gpu

    Returns:
        best_caption (List[str]), attention_weights (np.ndarray)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # reverse map idx to word
    rev_map = {idx: w for w, idx in word_map.items()}

    # preprocess and encode image
    image_tensor = _load_and_preprocess(image_path).unsqueeze(0).to(device)
    encoder_out = encoder(image_tensor)  # (1, grid, grid, enc_dim)
    grid_size, enc_dim = encoder_out.size(1), encoder_out.size(3)
    encoder_feats = encoder_out.view(1, -1, enc_dim)  # (1, num_pixels, enc_dim)

    # expand for beam
    num_pixels = encoder_feats.size(1)
    encoder_feats = encoder_feats.expand(beam_size, num_pixels, enc_dim)

    # Beam‐search initialization
    k = beam_size
    sequences = torch.full(
        (k, 1), word_map['<start>'], dtype=torch.long, device=device
    )
    scores = torch.zeros(k, 1, device=device)
    alphas = torch.ones(k, 1, grid_size, grid_size, device=device)

    completed_seq = []
    completed_scores = []
    completed_alphas = []

    # initialize LSTM state
    h, c = decoder.init_hidden_states(encoder_feats)

    step = 0
    while True:
        step += 1
        # embed last tokens
        last_tokens = sequences[:, -1]
        embeddings = decoder.embedding(last_tokens)

        # attention
        ctx, alpha = decoder.attention(encoder_feats, h)  
        alpha = alpha.view(-1, grid_size, grid_size)

        # gating
        beta = torch.sigmoid(decoder.beta_gate(h))
        ctx = beta * ctx

        # decode one step
        h, c = decoder.decoder_cell(
            torch.cat([embeddings, ctx], dim=1), (h, c)
        )                                               
        logits = decoder.output_proj(h)
        log_probs = F.log_softmax(logits, dim=1)

        # accumulate scores
        total_scores = scores + log_probs.unsqueeze(1)
        flat_scores = total_scores.view(-1)
        
        if step == 1:
            top_scores, top_indices = flat_scores.topk(k, dim=0)
        else:
            top_scores, top_indices = flat_scores.topk(k)

        prev_inds = top_indices // decoder.vocab_size
        next_inds = top_indices % decoder.vocab_size
        
        # build new candidates
        new_sequences = []
        new_alphas = []
        for beam_idx, token_idx in zip(prev_inds, next_inds):
            seq = sequences[beam_idx].tolist() + [token_idx.item()]
            new_sequences.append(seq)
            new_alphas.append(
                torch.cat([
                    alphas[beam_idx],
                    alpha[beam_idx].unsqueeze(0)
                ], dim=0)
            )
        sequences = torch.tensor(new_sequences, device=device)
        alphas = torch.stack(new_alphas, dim=0)
        scores = top_scores.unsqueeze(1)

        # check for completions
        incomplete = []
        for i, seq in enumerate(sequences):
            if seq[-1].item() == word_map['<end>']:
                completed_seq.append(seq)
                completed_scores.append(scores[i].item())
                completed_alphas.append(alphas[i].cpu().numpy())
            else:
                incomplete.append(i)

        # reduce beam
        k = len(incomplete)
        if k == 0 or step >= max_caption_len:
            break

        sequences = sequences[incomplete]
        alphas = alphas[incomplete]
        scores = scores[incomplete]
        h = h[prev_inds[incomplete]]
        c = c[prev_inds[incomplete]]
        encoder_feats = encoder_feats[prev_inds[incomplete]]

    # pick best
    best_idx = np.argmax(completed_scores)
    best_seq = completed_seq[best_idx]
    best_alphas = completed_alphas[best_idx]

    # map to words
    caption = [rev_map[token] for token in best_seq]

    if return_attention:
        _plot_attention(image_path, caption, best_alphas, grid_size)

    return caption, best_alphas
