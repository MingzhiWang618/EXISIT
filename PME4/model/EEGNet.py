import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/data2/mingzhi/BCI/WMZ_BCI')
import os
from typing import Optional


class EEGNet(nn.Module):
    def __init__(self, nb_classes, Chans=64, Samples=128, dropoutRate=0.5, 
                 kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()
        self.drop_rate = dropoutRate
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, D*F1, (Chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(D*F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_rate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(D*F1, F2, (1, 16), groups=D*F1, bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(self.drop_rate)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),  
            nn.Linear(208, nb_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


