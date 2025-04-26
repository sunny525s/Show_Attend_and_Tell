import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-params (must match training)
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

# Choose image and beam size
image_path = "/content/Flicker8k_Dataset/1030985833_b0902ea560.jpg"
beam_size  = 5

# Display Image
img = Image.open(image_path).convert("RGB")
plt.figure(figsize=(6,6))
plt.imshow(img)
plt.axis("off")
plt.title("Input Image")
plt.show()

# Run inference and plot attention
caption, alphas = generate_caption(
    encoder=encoder,
    decoder=decoder,
    image_path=image_path,
    word_map=word2id,
    beam_size=beam_size,
    return_attention=True,
    device=device,
)

# Print out the final caption
print("Predicted caption:")
print(" ".join(caption))
