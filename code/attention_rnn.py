import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # (batch_size, num_pixels, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # (batch_size, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)  # (batch_size, num_pixels)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # (batch_size, num_pixels)

    def forward(self, encoder_out, decoder_hidden):
        annotation_vec = self.encoder_att(encoder_out)  # project each image region feature a_i into the attention space
        hidden_state = self.decoder_att(decoder_hidden)  # projects decoder's hidden state to the attention space
        eti = self.full_att(self.relu(annotation_vec + hidden_state.unsqueeze(1))).squeeze(2)  # e_{ti}
        alpha = self.softmax(eti)  # softmax layer calculates attention weights
        context_vec = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # z_t the context vector

        return context_vec, alpha