import json
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import skimage.transform
from torchvision import transforms

from encoder import ImageEncoder
from decoder import Decoder

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word map
word_map_path = "/content/checkpoints/word_map.json"  # <-- your vocab file
with open(word_map_path, "r") as j:
    word2id = json.load(j)

# Hyperparameters (must match training)
embed_dim     = 512
attention_dim = 512
decoder_dim   = 512
encoder_dim   = 1920
vocab_size    = len(word2id)

# Instantiate models
encoder = ImageEncoder(output_size=14).to(device)
decoder = Decoder(
    attention_dim=attention_dim,
    embed_dim=embed_dim,
    decoder_dim=decoder_dim,
    vocab_size=vocab_size,
    encoder_dim=encoder_dim,
).to(device)

# Load checkpoint
ckpt_path = Path("/content/checkpoints/image_captioning_best.pth")
ckpt = torch.load(ckpt_path, map_location=device)
encoder.load_state_dict(ckpt["encoder_state_dict"])
decoder.load_state_dict(ckpt["decoder_state_dict"])
print(f"Loaded checkpoint from epoch {ckpt['epoch']}, BLEU-4={ckpt['bleu-4']:.4f}")

encoder.eval()
decoder.eval()

# Inference Utilities
def _load_and_preprocess(image_path: str, image_size: int = 256):
    img = np.array(Image.open(image_path).convert('RGB'))
    img = cv2.resize(img, (image_size, image_size))
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return tf(img)

def _plot_attention(image_path: str, caption: list, alphas: np.ndarray, grid_size: int, upsample: int = 24):
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
        alpha_up = skimage.transform.pyramid_expand(alpha_map, upscale=upsample, sigma=8)
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
    if '<end>' in word_map:
        end_idx = word_map['<end>']
    elif '<eos>' in word_map:
        end_idx = word_map['<eos>']
    elif 'endseq' in word_map:
        end_idx = word_map['endseq']
    else:
        raise KeyError(
            "No end-of-sentence token found in word_map."
        )

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    image = transform(img).unsqueeze(0).to(device)

    encoder_out = encoder(image)  # (1, H, W, C)
    encoder_out = encoder_out.permute(0, 3, 1, 2)  # (1, C, H, W)
    enc_dim = encoder_out.size(1)
    encoder_out = encoder_out.reshape(1, enc_dim, -1).permute(0, 2, 1)  # (1, num_pixels, enc_dim)
    num_pixels = encoder_out.size(1)

    encoder_out = encoder_out.expand(beam_size, num_pixels, enc_dim)
    k = beam_size
    vocab_size = len(word_map)
    seqs = [[word_map['<start>']]] * k
    top_k_scores = torch.zeros(k, 1).to(device)
    seqs_alpha = torch.ones(k, 1, num_pixels).to(device)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []

    h, c = decoder.init_hidden_states(encoder_out)

    step = 1
    while True:
        prev_words = torch.LongTensor([seq[-1] for seq in seqs]).to(device)
        embeddings = decoder.embedding(prev_words)

        context, alpha = decoder.attention(encoder_out, h)
        gate = torch.sigmoid(decoder.beta_gate(h))
        context = gate * context

        h, c = decoder.decoder_cell(torch.cat([embeddings, context], dim=1), (h, c))
        scores = decoder.output_proj(h)
        scores = F.log_softmax(scores, dim=1)

        scores = top_k_scores.expand_as(scores) + scores

        if step == 1:
            scores = scores[0]

        scores, indices = scores.view(-1).topk(k, dim=0, largest=True, sorted=True)
        prev_beams = indices // vocab_size
        next_words = indices % vocab_size

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
    caption = []
    for idx in best_seq:
        w = idx2word[idx]
        if w == '<start>':
            continue
        if idx == end_idx:
            break
        caption.append(w)

    return (caption, alphas) if return_attention else caption

# Choose Image
image_path = "/content/Flicker8k_Dataset/1030985833_b0902ea560.jpg"
beam_size = 5

# Show Image
img = Image.open(image_path).convert("RGB")
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Generate Caption
caption, alphas = generate_caption(
    encoder=encoder,
    decoder=decoder,
    image_path=image_path,
    word_map=word2id,
    beam_size=beam_size,
    return_attention=True,
    device=device,
)

# Print Caption
print("Predicted caption:")
print(" ".join(caption))

# Plot Attention
_plot_attention(
    image_path=image_path,
    caption=caption,
    alphas=alphas,
    grid_size=14
)
