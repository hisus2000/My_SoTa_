import sys
import pathlib


import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


ROOT_DIR = pathlib.Path(__file__).absolute().parent.parent
sys.path.append(str(ROOT_DIR))

from base import BaseModel
import torch.nn as nn


class Resnet50Model(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet50 = models.resnet50(pretrained= False)

        self.resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=128),
            nn.Dropout(0.2, inplace=True),
            nn.Linear(in_features=128, out_features=num_classes)
        )
        # freeze = False, unfreeze = False
        for param in self.resnet50.parameters():
            param.require_grad = True

    def forward(self, x):
        x = self.resnet50(x)
        return x
