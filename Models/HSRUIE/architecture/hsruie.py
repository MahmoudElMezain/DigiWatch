"""Run the HSRUIE evaluation forward pass using its VGG features, refinement networks, and semantic decoder."""
from __future__ import annotations

from torch import nn

from models.DenseNet import FE, SAD
from models.FEM import FEM, MIDDLE


class HSRUIE(nn.Module):
    def __init__(self, vgg_features):
        """Pretrained VGG feature modules -> HSRUIE inference network."""
        super().__init__()
        self.slices = nn.ModuleList([
            nn.Sequential(*[vgg_features[index] for index in range(start, end)])
            for start, end in ((0, 4), (4, 9), (9, 14))
        ])
        self.nc = FE()
        self.high2 = MIDDLE()
        self.high3 = FEM(256, 256)
        self.sad = SAD()

    def forward(self, tensor):
        """Normalized RGB tensors -> normalized enhanced RGB tensors."""
        features = []
        current = tensor
        for block in self.slices:
            current = block(current)
            features.append(current)
        low, middle, high = self.nc(features)
        return self.sad([low, self.high2(tensor, middle), self.high3(high)])
