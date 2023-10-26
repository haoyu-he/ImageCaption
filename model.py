import torch
import torch._utils
import torch.nn as nn
import torchvision.models as models

from typing import Tuple


class Encoder(nn.Module):

    def __init__(self, img_emb_dim: int):
        super().__init__()

        self.img_emb_dim = img_emb_dim

        # freeze encoder parameters
        encoder = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in encoder.parameters():
            param.requires_grad_(False)

        # remove the last layer
        modules = list(encoder.children())[:-1]
        self.encoder = nn.Sequential(*modules).to(self.device)

        # final layer
        self.fc = nn.Linear(encoder.fc.in_features, self.img_emb_dim, device=self.device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:

        h = self.encoder(images)
        h = h.reshape(h.shape[0], -1)
        # h: (batch, 2048)

        h = self.fc(h)
        # h: (batch, img_emb_dim)

        return h


class Decoder(nn.Module):

    def __init__(self,
                 img_emb_dim: int,
                 word_emb_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 vocab_size: int):
        super().__init__()

        self.img_emb_dim = img_emb_dim
        self.word_emb_dim = word_emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        self.decoder = nn.LSTM(input_size=self.img_emb_dim + self.word_emb_dim,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.vocab_size),
            nn.LogSoftmax(dim=2)
        )

    def forward(self,
                embedded_captions: torch.Tensor,
                encoder_emb: torch.Tensor,
                hidden: torch.Tensor,
                cell: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        decoder_input = torch.cat((embedded_captions, encoder_emb), dim=2)

        decoder_output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        # decoder_output: (length, batch, hidden_dim)

        decoder_output = self.fc(decoder_output)
        # decoder_output: (length, batch, vocab_size)

        return decoder_output, (hidden, cell)
