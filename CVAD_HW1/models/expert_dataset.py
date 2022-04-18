from sklearn import datasets
from torch.utils.data import Dataset
import torch
import os
from torchvision.io import read_image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import json
from torchvision import datasets, models, transforms

class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""
    def __init__(self, data_root, transform, modelType):
        self.inputs = {}
        self.transform = transform
        self.data_root = data_root
        self.modelType = modelType
        self.imageDir = os.path.join(self.data_root, "rgb")
        self.imageNames = os.listdir(self.imageDir) 
        self.imageNames.sort()

        self.labelDir = os.path.join(self.data_root, "measurements")
        self.labelNames = os.listdir(self.labelDir)
        self.labelNames.sort()
    def __len__(self):
        return len(self.imageNames)
    def __getitem__(self, index):
        """Return RGB images and measurements"""
        imageName = self.imageNames[index]
        imagePath = os.path.join(self.data_root, 'rgb', imageName)
        image = Image.open(imagePath)

        if self.transform:
            image = self.transform(image)

        labelName = self.labelNames[index]
        labelPath = os.path.join(self.data_root, 'measurements', labelName)
        jsonData = json.load(open(labelPath))

        speed = torch.tensor(jsonData["speed"])
        command = torch.tensor(jsonData["command"])
        throttle = torch.tensor(jsonData["throttle"])
        brake = torch.tensor(jsonData["brake"])
        steer = torch.tensor(jsonData["steer"])
        traffLD = torch.tensor(jsonData["tl_dist"])
        traffLS = torch.tensor(jsonData["tl_state"])
        laneDist = torch.tensor(jsonData["lane_dist"])
        routeAngle = torch.tensor(jsonData["route_angle"])
        
        if self.modelType == "CILRS":
            return image, speed, command, throttle, brake, steer
        else:
            return image, command, traffLD, traffLS, laneDist, routeAngle

        