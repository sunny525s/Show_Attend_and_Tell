import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        # (batch_size, num_pixels, attention_dim)
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        # (batch_size, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        # (batch_size, num_pixels)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()
        # (batch_size, num_pixels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # project each image region feature a_i into the attention space
        annotation_vec = self.encoder_att(encoder_out)
        # projects decoder's hidden state to the attention space
        hidden_state = self.decoder_att(decoder_hidden)
        # e_{ti}
        eti = self.full_att(
            self.tanh(annotation_vec + hidden_state.unsqueeze(1))
        ).squeeze(2)
        # softmax layer calculates attention weights
        alpha = self.softmax(eti)
        # z_t the context vector
        context_vec = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context_vec, alpha
