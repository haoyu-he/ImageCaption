import math

import torch
import torch.nn as nn
import torchvision.models as models

from typing import Tuple


class Encoder(nn.Module):

    def __init__(self, word_emb_dim: int):
        super().__init__()

        self.word_emb_dim = word_emb_dim

        # freeze encoder parameters
        encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in encoder.parameters():
            param.requires_grad_(False)

        # remove the last layer
        modules = list(encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        # final layer
        self.fc = nn.Linear(encoder.fc.in_features, self.word_emb_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        h = self.encoder(images)
        h = h.reshape(h.shape[0], -1)
        # h: (batch, 2048)

        h = self.fc(h)
        # h: (batch, img_emb_dim)

        return h


class DecoderLSTM(nn.Module):

    def __init__(self,
                 word_emb_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 vocab_size: int):
        super().__init__()

        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.hidden_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim, requires_grad=True))
        self.cell_0 = nn.Parameter(torch.zeros(self.num_layers, 1, self.hidden_dim, requires_grad=True))

        self.decoder = nn.LSTM(input_size=self.word_emb_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self,
                decoder_input: torch.Tensor,
                hidden: torch.Tensor,
                cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        # decoder_output: (length, batch, hidden_dim)

        decoder_output = self.fc(decoder_output)
        # decoder_output: (length, batch, vocab_size)

        return decoder_output, (hidden, cell)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DecoderGPT1(nn.Module):

    def __init__(self,
                 word_emb_dim: int,
                 nhead: int,
                 hidden_dim: int,
                 num_layers: int,
                 vocab_size: int):
        super().__init__()

        self.word_emb_dim = word_emb_dim
        self.nhead = nhead
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.pe = PositionalEncoding(self.word_emb_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.word_emb_dim,
            nhead=self.nhead,
            dim_feedforward=self.hidden_dim
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.word_emb_dim, self.vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self, decoder_input: torch.Tensor):

        src = self.pe(decoder_input)

        mask = nn.Transformer.generate_square_subsequent_mask(src.shape[0], device=src.device)

        decoder_output = self.decoder(src, mask=mask)
        decoder_output = self.fc(decoder_output)
        # decoder_output: (length, batch, vocab_size)

        return decoder_output