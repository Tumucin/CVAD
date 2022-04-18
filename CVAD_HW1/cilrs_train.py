import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
from models.fc import FC
from models.branching import BRANCHING
from models.join import JOIN
from torch.utils.data._utils.collate import default_convert
import numpy as np
from torch import optim
import matplotlib.pyplot as plt
import copy
import os
def validate(model, val_loader, batch_size):
    """Validate model performance on the validation dataset"""
    batchLoss = 0
    epochLoss = 0
    numOfSample = 10000
    batch_size = 128
    num_of_batches = numOfSample / batch_size 
    with torch.no_grad():
        for batch, (image, speed, command, throttle, brake, steer) in enumerate(val_loader):
            image, speed, command = image.type(torch.cuda.FloatTensor), speed.type(torch.cuda.FloatTensor)\
                                        ,command.type(torch.cuda.LongTensor)
            speed = speed.view((batch_size, 1))
            command = command.view((batch_size, 1))
            # Forward propagation
            branchesOutputs, speedBranchOutput = model(image, speed, command)
            groundTruthBranches = torch.stack([throttle,brake,steer],dim=1).cuda()
            branchesOutputs = torch.stack([branchesOutputs[0],branchesOutputs[1],
                                branchesOutputs[2], branchesOutputs[3]], dim=1)
            command = command.view((batch_size, 1))
            controls_masks = loss_mask(command)
            loss = calculateLoss(branchesOutputs, groundTruthBranches,controls_masks, \
                                        speed, speedBranchOutput)
            batchLoss += loss.item()
            epochLoss += loss.item()
            if (batch +1) % 50 == 0:    
                print('Batch: {}, Avg. Loss: {}'.format(batch+1, batchLoss/50))
                batchLoss = 0.0    
        
                
    print('----------------\nEpoch loss: {}\n------------\n'.format(epochLoss/num_of_batches))
    return epochLoss


def train(model, train_loader, batch_size, opt):
    """Train model on the training dataset for one epoch"""
    # Your code here
    batchLoss = 0
    epochLoss = 0
    
    for batch, (image, speed, command, throttle, brake, steer) in enumerate(train_loader):
        #print(f"Batch {batch+1}\n-------------------------------")
        size = len(train_loader.dataset)
        num_of_batches = size / batch_size     
        image, speed, command = image.type(torch.cuda.FloatTensor), speed.type(torch.cuda.FloatTensor)\
                                    ,command.type(torch.cuda.LongTensor)
        speed = speed.view((batch_size, 1))
        command = command.view((batch_size, 1))
        # Forward propagation
        branchesOutputs, speedBranchOutput = model(image, speed, command)
        groundTruthBranches = torch.stack([throttle,brake,steer],dim=1).cuda()
        branchesOutputs = torch.stack([branchesOutputs[0],branchesOutputs[1],
                            branchesOutputs[2], branchesOutputs[3]], dim=1)
        command = command.view((batch_size, 1))
        controls_masks = loss_mask(command)
        loss = calculateLoss(branchesOutputs, groundTruthBranches,controls_masks, \
                                    speed, speedBranchOutput)  
        #a = list(model.parameters())[1].clone()
        opt.zero_grad()
        loss.backward()
        opt.step()
        #b = list(model.parameters())[1].clone()
        #print(torch.equal(a.data, b.data))
        batchLoss += loss.item()
        epochLoss += loss.item()
        if (batch +1) % 50 == 0:    
            current = (batch + 1) * batch_size
            print(f"[{current:>5d}/{size:>5d}]")
            print('Batch: {}, Avg. Loss: {}'.format(batch+1, batchLoss/50))
            batchLoss = 0.0    
        
                
    print('----------------\nEpoch loss: {}\n------------\n'.format(epochLoss/num_of_batches))
    return epochLoss

def calculateLoss(branchesOutputs, groundTruthBranches, \
                    controls_masks, speed, speedBranchOutput):
    branchesLoss = []
    for i in range(4):
        branchesLoss.append(torch.abs((branchesOutputs[:, i, :]  \
                               - groundTruthBranches)*controls_masks[i]))

    branchesLoss.append(torch.abs(speed - speedBranchOutput))

    for i in range(4):
        branchesLoss[i] = branchesLoss[i][:, 0] + branchesLoss[i][:, 1] \
                        + branchesLoss[i][:, 2]
    lossBranches = branchesLoss[0] + branchesLoss[1] + branchesLoss[2] + branchesLoss[3]
    
    finalLoss = (torch.sum(lossBranches) + torch.sum(branchesLoss[4])) / 3

    return finalLoss
    
def loss_mask(command):
    controls_masks = []
    numOfActions = 3
    for i in range(4):
        controls = (command == i)
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = torch.cat([controls] * numOfActions, 1)
        controls_masks.append(controls)
    return controls_masks

def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss)
    plt.plot(np.linspace(1, len(val_loss), len(val_loss)), val_loss )
    plt.legend(['Train Loss [Epoch]', 'Validation Loss [Epoch]'])
    plt.show()

def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainFileName = 'expert_data/train'
    valFileName = 'expert_data/val'
    train_root = os.path.join(os.getcwd(), trainFileName)
    val_root = os.path.join(os.getcwd(), valFileName)
    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    model = CILRS().cuda()
    #model.to(device)
    model.train()
    train_dataset = ExpertDataset(train_root, transform, "CILRS")
    val_dataset = ExpertDataset(val_root, transform, "CILRS")

    # You can change these hyper parameters freely, and you can add more
    opt = optim.SGD(model.parameters(),lr=0.01,momentum=0.9,nesterov=True)
    num_epochs = 5
    batch_size = 128 # Normally it is 64

    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                drop_last=True)

    train_losses = []
    val_losses = []
    
    for i in range(num_epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        #a = list(model.parameters())[0].clone()
        train_losses.append(train(model, train_loader, batch_size, opt))
        #b = list(model.parameters())[0].clone()
        #print(torch.equal(a.data, b.data))
        
        val_losses.append(validate(model, val_loader, batch_size))
    print("Done!")
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
