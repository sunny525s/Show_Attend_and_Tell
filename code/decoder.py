import torch
import torch.nn as nn

from attention import Attention


class Decoder(nn.Module):
    """
    LSTM decoder with attention for image captioning.
    """

    def __init__(
        self,
        attention_dim: int,
        embed_dim: int,
        decoder_dim: int,
        vocab_size: int,
        encoder_dim: int = 2048,
        dropout: float = 0.5,
    ):
        super().__init__()
        # feature sizes
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # modules
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.decoder_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # initialize LSTM from image features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # gating for context vector
        self.beta_gate = nn.Linear(decoder_dim, encoder_dim)

        self.output_proj = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.output_proj.weight.data.uniform_(-0.1, 0.1)
        self.output_proj.bias.data.zero_()

    def init_hidden_states(self, encoder_feats: torch.Tensor):
        """
        Create initial hidden & cell states from the mean encoder features.
        :param encoder_feats: (batch, num_pixels, encoder_dim)
        """
        mean_feats = encoder_feats.mean(dim=1)
        return self.init_h(mean_feats), self.init_c(mean_feats)

    def forward(
        self,
        encoder_out: torch.Tensor,
        captions: torch.Tensor,
        caption_lengths: torch.Tensor,
    ):
        """
        :param encoder_out: (batch, enc_size, enc_size, encoder_dim)
        :param captions:    (batch, max_caption_len)
        :param caption_lengths: (batch, 1)
        :returns:
          - predictions:      (batch, max_steps, vocab_size)
          - sorted_captions:  (batch, max_caption_len)
          - decode_lengths:   List[int]
          - attention_weights:(batch, max_steps, num_pixels)
          - sort_idx:         (batch,)
        """
        batch_size = encoder_out.size(0)
        # (batch, num_pixels, encoder_dim)
        encoder_feats = encoder_out.view(batch_size, -1, self.encoder_dim)
        num_pixels = encoder_feats.size(1)

        # descending caption length
        lengths, sort_idx = caption_lengths.squeeze(1).sort(descending=True)
        encoder_feats = encoder_feats[sort_idx]
        sorted_caps = captions[sort_idx]
        decode_lengths = (lengths - 1).tolist()
        max_steps = max(decode_lengths)

        embed = self.embedding(sorted_caps)  # (batch, max_len, embed_dim)

        # init LSTM state
        hidden_state, cell_state = self.init_hidden_states(encoder_feats)

        device = encoder_feats.device
        preds = encoder_feats.new_zeros(batch_size, max_steps, self.vocab_size)
        alphas = encoder_feats.new_zeros(batch_size, max_steps, num_pixels)

        for t in range(max_steps):
            active_mask = torch.tensor([l > t for l in decode_lengths], device=device)
            if not active_mask.any():
                break
            idx = active_mask.nonzero(as_tuple=False).squeeze(1)

            # attend
            ctx, alpha = self.attention(encoder_feats[idx], hidden_state[idx])

            # gate the context
            gate = torch.sigmoid(self.beta_gate(hidden_state[idx]))
            ctx = gate * ctx

            # LSTMCell
            lstm_input = torch.cat([embed[idx, t, :], ctx], dim=1)
            h_t, c_t = self.decoder_cell(
                lstm_input, (hidden_state[idx], cell_state[idx])
            )
            hidden_state[idx], cell_state[idx] = h_t, c_t

            # project to vocab
            step_preds = self.output_proj(self.dropout(h_t))
            preds[idx, t, :] = step_preds
            alphas[idx, t, :] = alpha.to(alphas.dtype)

        return preds, sorted_caps, decode_lengths, alphas, sort_idx
