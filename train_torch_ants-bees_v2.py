# Ants and Bees 
# Author: Morpheus Hsieh (morpheus.hsieh@gmail.com)

from __future__ import print_function, division  

import os, sys
import argparse
import copy
import matplotlib.pyplot as plt
import numpy as np
import shutil
import time
import zipfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Subset, DataLoader

import torchvision
from torchvision import datasets, models, transforms

from configs.config_train import get_cfg_defaults


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


# Model finetune
def finetuneModel(class_names):
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features 
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))
    return model_ft


def train(model, loader, device, optimizer, criterion, datasize, scheduler):
    model.train()
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
            
        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()
            
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    scheduler.step()
        
    epoch_loss = running_loss / datasize
    epoch_acc = running_corrects.double() / datasize
        
    print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return


def valid(model, loader, device, optimizer, criterion, datasize, best_model_wts, best_acc):
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
        
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / datasize
    epoch_acc = running_corrects.double() / datasize
        
    print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        
    # deep copy the model
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        
    return best_acc, best_model_wts  


def main():
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    args = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print('\n', cfg)

    RawPath = cfg.DATA.RAW_PATH
    ProcPath = cfg.DATA.PROCESSED_PATH

    # Normalize
    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    # Datasets
    trainSet = datasets.ImageFolder(os.path.join(ProcPath, 'train'), train_transform)
    validSet = datasets.ImageFolder(os.path.join(ProcPath, 'valid'), valid_transform)

    ClassNames = trainSet.classes
    print('\nTrain classes: ', ClassNames)

    TrainSize = len(trainSet)
    ValidSize = len(validSet)
    print('\nTrainSet size: ', TrainSize)
    print('ValidSet size: ', ValidSize)

    # DataLoaders
    BatchSize = cfg.TRAIN.BATCH_SIZE
    NumWorkers = cfg.TRAIN.NUM_WORKERS

    trainLoader = DataLoader(
        trainSet, batch_size=BatchSize, shuffle=True,  num_workers=NumWorkers
    )

    validLoader = DataLoader(
        validSet, batch_size=BatchSize, shuffle=False, num_workers=NumWorkers
    )

    images, classes = next(iter(trainLoader))
    print('\nTrainLoader images shape: ', images.shape)
    print('TrainLoader classes shape: ', classes.shape)

    # GPU Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('\nDevice: ', device)

    # Model Finetune
    model_ft = finetuneModel(ClassNames).to(device)
    print('\n', model_ft)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    lr = cfg.TRAIN.LEARNING_RATE
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Training and Validate model
    since = time.time()
    print('\nTraining start...')

    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0
    
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 30)
    
        train(
            model_ft, trainLoader, device, optimizer_ft, criterion, TrainSize, 
            exp_lr_scheduler
        )
    
        best_acc, best_model_wts = valid(
            model_ft, validLoader, device, optimizer_ft, criterion, ValidSize, 
            best_model_wts, best_acc
        )
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    print(best_model_wts)



    return 


if __name__=='__main__':
    main()
