import torch
import torch.nn as nn
import torchvision.models as models

from typing import Tuple


class ImageEncoder(nn.Module):

    def __init__(self,
                 word_emb_dim: int,
                 encoder_feedforward_dim: int,
                 encoder_nheads: int,
                 num_encoder_layer: int):
        super().__init__()

        self.image_emb_dim = word_emb_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.encoder_nheads = encoder_nheads
        self.num_encoder_layer = num_encoder_layer

        # freeze encoder parameters
        encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in encoder.parameters():
            param.requires_grad_(False)

        # remove the last layer
        modules = list(encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        # transformer mapping
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder.fc.in_features,
            nhead=self.encoder_nheads,
            dim_feedforward=self.encoder_feedforward_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, self.num_encoder_layer)

        # final layer
        self.fc = nn.Linear(encoder.fc.in_features, self.image_emb_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        h = self.encoder(images)
        h = h.reshape(h.shape[0], -1)
        # h: (batch, 2048)

        h = self.fc(h)
        # h: (batch, img_emb_dim)

        return h


class TextDecoder(nn.Module):

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
