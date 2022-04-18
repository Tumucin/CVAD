import torch.nn as nn
from torchvision import models
from torchvision import datasets, models, transforms
from expert_dataset import ExpertDataset
from torch.utils.data import DataLoader
from models.fc import FC
from models.branching import BRANCHING
class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""
    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.perception = models.resnet18(pretrained=True)
        self.perception.fc = nn.Linear(512, 512)

        self.blockParams = [(512,128), (128, 128),
                                 (128,1)]
        self.block = FC(self.blockParams)   

        self.blockCmdParams = [(512,128), (128, 128),
                                 (128,1)]
        branchVector = []

        for i in range(4):
            branchVector.append(FC(self.blockCmdParams))
        
        self.branches = BRANCHING(branchVector)
    def forward(self, img):
        percepOutput = self.perception(img)
        prediction = {}
        prediction['trafficLD']    = self.block(percepOutput)
        prediction['trafficLS']    = self.block(percepOutput)
        prediction['laneDistance'] = self.branches(percepOutput)
        prediction['routeAngle']   = self.branches(percepOutput)

        return prediction

