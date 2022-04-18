from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from PIL import Image
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
from expert_dataset import ExpertDataset
from torch.utils.data import DataLoader

class BRANCHING(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self, branchVector):
        super(BRANCHING, self).__init__()
        self.branchModules = nn.ModuleList(branchVector)


    def forward(self, x):
        branchOutputs = []

        for branch in self.branchModules:
            branchOutputs.append(branch(x))
        return branchOutputs
