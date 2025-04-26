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
    encoder,
    decoder,
    image_path: str,
    word_map: dict,
    beam_size: int = 5,
    max_caption_len: int = 50,
    return_attention: bool = False,
    device: str = "cpu"
):
    """
    Generate a caption for an image using beam search.
    """

    if '<end>' in word_map:
        end_idx = word_map['<end>']
    elif '<eos>' in word_map:
        end_idx = word_map['<eos>']
    elif 'endseq' in word_map:
        end_idx = word_map['endseq']
    else:
        raise KeyError(
            "No end-of-sentence token found in word_map. "
            "Make sure one of '<end>', '<eos>', or 'endseq' is present."
        )

    # Load & preprocess image
    img = Image.open(image_path).convert("RGB")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        normalize
    ])
    image = transform(img).unsqueeze(0).to(device) # (1,3,256,256)

    # Encode
    encoder_out = encoder(image) # (1, H, W, C)
    encoder_out = encoder_out.permute(0, 3, 1, 2) # (1, C, H, W)
    enc_dim     = encoder_out.size(1)
    encoder_out = encoder_out.reshape(1, enc_dim, -1).permute(0,2,1) # (1, num_pixels, enc_dim)
    num_pixels  = encoder_out.size(1)

    # Beam search
    encoder_out = encoder_out.expand(beam_size, num_pixels, enc_dim)
    k = beam_size
    vocab_size = len(word_map)
    seqs = [[word_map['<start>']]] * k
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, num_pixels).to(device)

    complete_seqs = []
    complete_seqs_alpha  = []
    complete_seqs_scores = []

    # hidden & cell from decoder’s init method
    h, c = decoder.init_hidden_states(encoder_out)

    step = 1
    while True:
        # last words
        prev_words = torch.LongTensor([seq[-1] for seq in seqs]).to(device)
        embeddings = decoder.embedding(prev_words)

        # attention
        context, alpha = decoder.attention(encoder_out, h)
        gate = torch.sigmoid(decoder.beta_gate(h))
        context = gate * context

        # decode one step
        h, c = decoder.decoder_cell(
            torch.cat([embeddings, context], dim=1), (h, c)
        )
        scores = decoder.output_proj(h)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            scores = scores[0]

        # flatten & top-k
        scores, indices = scores.view(-1).topk(
            k, dim=0, largest=True, sorted=True
        )
        prev_beams = indices // vocab_size
        next_words = indices %  vocab_size

        # build new beams
        new_seqs = []
        new_seqs_alpha = []
        for b, w in zip(prev_beams, next_words):
            new_seqs.append(seqs[b] + [w.item()])
            new_seqs_alpha.append(
                torch.cat([seqs_alpha[b], alpha[b].unsqueeze(0)], dim=0)
            )

        seqs = []
        seqs_alpha = []
        for i, seq in enumerate(new_seqs):
            if seq[-1] == end_idx:
                complete_seqs.append(seq)
                complete_seqs_alpha.append(new_seqs_alpha[i])
                complete_seqs_scores.append(scores[i])
            else:
                seqs.append(seq)
                seqs_alpha.append(new_seqs_alpha[i])

        k = len(seqs)
        if k == 0 or step >= max_caption_len:
            break

        h = h[prev_beams[:k]]
        c = c[prev_beams[:k]]
        encoder_out = encoder_out[prev_beams[:k]]
        top_k_scores = scores[:k].unsqueeze(1)
        step += 1

    if not complete_seqs_scores:
        complete_seqs = seqs
        complete_seqs_alpha = seqs_alpha
        complete_seqs_scores = [s.item() for s in top_k_scores.squeeze(1)]

    best_idx = complete_seqs_scores.index(max(complete_seqs_scores))
    best_seq = complete_seqs[best_idx]
    alphas = complete_seqs_alpha[best_idx].cpu().detach().numpy()

    idx2word = {v: k for k, v in word_map.items()}
    caption  = []
    for idx in best_seq:
        w = idx2word[idx]
        if w == '<start>': continue
        if idx == end_idx:   break
        caption.append(w)

    return (caption, alphas) if return_attention else caption
