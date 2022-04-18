from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from tkinter import X
from PIL import Image
from argon2 import PasswordHasher
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
from expert_dataset import ExpertDataset
from torch.utils.data import DataLoader

class FC(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self, params):
        super(FC, self).__init__()
        self.layers = []
        self.params = params
        for i in range(len(self.params)):
            fc = nn.Linear(self.params[i][0], self.params[i][1])
            relu = nn.ReLU()

            self.layers.append(nn.Sequential(*[fc, relu]))
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)