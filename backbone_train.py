from U_Net import u_net_backbone
from MODA_dataloader import MODA_proc
from dice_loss import GeneralizedDiceLoss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Loading data, setting up GPU use, setting up variables for model training
BATCH_SIZE = 64
EPOCHS = 100


dataset_train = MODA_proc(input_path = '/scratch/s174411/FIL_CEN/TRAIN/input/', label_path = '/scratch/s174411/FIL_CEN/TRAIN/labels/')
dataset_val = MODA_proc(input_path = '/scratch/s174411/FIL_CEN/VAL/input/', label_path = '/scratch/s174411/FIL_CEN/VAL/labels/')

data_loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
data_loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)




net = u_net_backbone()
if torch.cuda.is_available():
    device = "cuda:0"
    if torch.cuda.device_count() > 1:
        print(torch.cuda.device_count(), " GPU's GIVEN")
        net = nn.DataParallel(net)
    else:
        print(device)
    net.to(device)

criterion = GeneralizedDiceLoss()
optimizer = optim.Adam(net.parameters(), lr=5.0e-3)

training_loss = []
validation_loss = []
for j, epoch in enumerate(range(EPOCHS)):  # loop over the dataset multiple times
    net.train()

    running_loss = []
    for i, batch in enumerate(data_loader_train):
        
            
        food, labels = batch
        food = food.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(food)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())

    print("Epoch", j, " Training loss: ", sum(running_loss)/len(running_loss))

    training_loss.append(sum(running_loss)/len(running_loss))
        

    net.eval()
    running_loss = []
    for i, batch in enumerate(data_loader_train): 
        food, labels = batch
        food = food.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = net(food)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())


    print("Epoch", j, " Validation loss: ", sum(running_loss)/len(running_loss))
    
    validation_loss.append(sum(running_loss)/len(running_loss))

#torch.save(net, '/home/marius/Documents/OneDrive/MSc/StartUP/Code/m1_stats_features.pt')
print('Finished Training')