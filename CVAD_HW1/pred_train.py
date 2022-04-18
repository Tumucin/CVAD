from numpy import dtype
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor
from torch import optim
import matplotlib.pyplot as plt
import numpy as np

def validate(model, val_loader, batch_size):
    batchLoss = 0
    epochLoss = 0
    size = len(val_loader.dataset)
    num_of_batches = size / batch_size
    with torch.no_grad():
        for batch, (image, command, traffLD, traffLS, laneDist, routeAngle) in enumerate(val_loader):
            image, command, traffLD = image.type(torch.cuda.FloatTensor), command.type(torch.cuda.LongTensor), traffLD.type(torch.cuda.LongTensor)
            traffLS, laneDist, routeAngle = traffLS.type(torch.cuda.FloatTensor), laneDist.type(torch.cuda.LongTensor), routeAngle.type(torch.cuda.LongTensor)

            prediction = model(image)
            command = command.view((batch_size, 1))
            controls_masks = loss_mask(command)
            loss = calculateLoss(laneDist, routeAngle, traffLD, traffLS, prediction, 
                        controls_masks, batch_size)
            batchLoss += loss.item()
            epochLoss += loss.item()
            if (batch +1) % 50 == 0:    
                print('Batch: {}, Avg. LossVal: {}'.format(batch+1, batchLoss/50))
                batchLoss = 0.0  
    print('----------------\nEpochVal loss: {}\n------------\n'.format(epochLoss/num_of_batches))

    return epochLoss/num_of_batches


def train(model, train_loader, batch_size, opt):
    """Train model on the training dataset for one epoch"""
    batchLoss = 0
    epochLoss = 0
    
    for batch, (image, command, traffLD, traffLS, laneDist, routeAngle) in enumerate(train_loader):
        #print(f"Batch {batch+1}\n-------------------------------")
        size = len(train_loader.dataset)
        num_of_batches = size / batch_size
        image, command, traffLD = image.type(torch.cuda.FloatTensor), command.type(torch.cuda.LongTensor), traffLD.type(torch.cuda.LongTensor)
        traffLS, laneDist, routeAngle = traffLS.type(torch.cuda.FloatTensor), laneDist.type(torch.cuda.LongTensor), routeAngle.type(torch.cuda.LongTensor)

        prediction = model(image)
        command = command.view((batch_size, 1))
        controls_masks = loss_mask(command)
        loss = calculateLoss(laneDist, routeAngle, traffLD, traffLS, prediction, 
                    controls_masks, batch_size)

        opt.zero_grad()
        loss.backward()
        opt.step()
        batchLoss += loss.item()
        epochLoss += loss.item()
        if (batch +1) % 50 == 0:    
            current = (batch + 1) * batch_size
            print(f"[{current:>5d}/{size:>5d}]")
            print('Batch: {}, Avg. LossTrain: {}'.format(batch+1, batchLoss/50))
            batchLoss = 0.0  
            
            
    print('----------------\nEpochTrain loss: {}\n------------\n'.format(epochLoss/num_of_batches))
    return epochLoss/num_of_batches

def loss_mask(command):
    controls_masks = []
    numOfActions = 1
    for i in range(4):
        controls = (command == i)
        controls = torch.tensor(controls, dtype=torch.float32)
        controls = torch.cat([controls] * numOfActions, 1)
        controls_masks.append(controls)
    return controls_masks        

def calculateLoss(laneDist, routeAngle, traffLD, traffLS,
                    prediction, controls_masks, batch_size):
    laneDistPredict = prediction['laneDistance']

    totalLoss = calBranchLoss(laneDistPredict, laneDist, 
                                controls_masks, batch_size)
    
    routeAnglePredict = prediction['routeAngle']
    totalLoss = totalLoss + calBranchLoss(routeAnglePredict, routeAngle, 
                                controls_masks, batch_size)
    traffLD = traffLD.view(batch_size, 1)
    totalLoss = totalLoss + torch.sum(torch.abs(prediction['trafficLD'] - traffLD))
    traffLS = traffLS.view(batch_size, 1)

    crossLoss = torch.nn.CrossEntropyLoss()
    traffLS = traffLS.float()
    
    totalLoss = totalLoss + crossLoss(prediction['trafficLS'], traffLS)
    return totalLoss
def calBranchLoss(prediction, groundTruth, controls_masks, batch_size):
    branchesOutputs = torch.stack([prediction[0],prediction[1],
                            prediction[2], prediction[3]], dim=1)
    branchesLoss = []
    groundTruth = groundTruth.view((batch_size, 1))
    for i in range(4):
        branchesLoss.append(torch.abs((branchesOutputs[:,i,:] - groundTruth)*controls_masks[i]))

    lossBranches = branchesLoss[0] + branchesLoss[1] + branchesLoss[2] + branchesLoss[3]
    lossBranches = torch.sum(lossBranches)

    return lossBranches
def plot_losses(train_loss, val_loss):
    plt.plot(np.linspace(1, len(train_loss), len(train_loss)), train_loss)
    plt.plot(np.linspace(1, len(val_loss), len(val_loss)), val_loss )
    plt.legend(['Train Loss [Epoch]', 'Validation Loss [Epoch]'])
    plt.show()
    


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = 'expert_data/train'
    val_root = 'expert_data/val'
    model = AffordancePredictor().cuda()
    model.train()
    transform = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ])
    train_dataset = ExpertDataset(train_root, transform, 'CAL')
    val_dataset = ExpertDataset(val_root, transform, 'CAL')

    # You can change these hyper parameters freely, and you can add more
    opt = optim.Adam(model.parameters(), lr = 0.00003)
    num_epochs = 7
    batch_size = 128
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    train_losses = []
    val_losses = []

    for i in range(num_epochs):
        train_losses.append(train(model, train_loader, batch_size, opt))
        val_losses.append(validate(model, val_loader, batch_size))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
