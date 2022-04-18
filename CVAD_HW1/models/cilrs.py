from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_STATUS_RESPONSE
from PIL import Image
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch
from expert_dataset import ExpertDataset
from torch.utils.data import DataLoader
from models.fc import FC
from models.branching import BRANCHING
from models.join import JOIN

class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""
    def __init__(self):
        super(CILRS, self).__init__()
        self.perception = models.resnet18(pretrained=True)
        self.perception.fc = nn.Linear(512, 512)
        self.measurementParams = [(1,128), (128, 128),
                                 (128,128)]
        self.measurement = FC(self.measurementParams)

        self.join = JOIN()
        self.speedPredParams = [(512,256), (256, 256),
                                 (256,1)]
        
        # For speed prediction
        self.speedBranch = FC(self.speedPredParams)

        branchVector = []
        self.branchParams = [(512,256), (256, 256),
                                 (256,3)]
        for i in range(4):
            branchVector.append(FC(self.branchParams))

        self.branches = BRANCHING(branchVector)
    def forward(self, img, measuredSpeed, command):
        percepOutput = self.perception(img)
        measurementOutput = self.measurement(measuredSpeed)
        joinedOutput = self.join(percepOutput, measurementOutput)
        speedBranchOutput = self.speedBranch(percepOutput)
        branchesOutputs = self.branches(joinedOutput)
        return branchesOutputs, speedBranchOutput
