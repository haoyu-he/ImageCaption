import torch
import torch._utils
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self, img_emb_dim, device):

        super(Encoder, self).__init__()
        self.img_emb_dim = img_emb_dim
        self.device = device

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

        images = images.to(self.device)

        h = self.encoder(images)
        h = h.reshape(h.shape[0], -1)
        # h: (batch, 2048)

        h = self.fc(h)
        # h: (batch, img_emb_dim)

        return h
