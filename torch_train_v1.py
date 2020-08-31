# Author: Morpehus Hsieh (morpheus.hsieh@gmail.com)

from __future__ import print_function, division

import os, sys
from configs.config_train import get_cfg_defaults

import argparse
import copy
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def parse_args():
    parser = argparse.ArgumentParser(description='Ants and Bees by PyTorch')
    parser.add_argument("--cfg", type=str, default="configs/config_train.yaml",
                        help="Configuration filename.")
    return parser.parse_args()


def train_model(dataloaders, device, dataset_sizes, 
    model, criterion, optimizer, scheduler, num_epochs=25):

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    args = parse_args()
    print(args)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print('\n', cfg)

    trans_mean = [0.485, 0.456, 0.406]
    trans_std  = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(
        mean = trans_mean,
        std  = trans_std
    )

    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Datasets
    proc_path = cfg.DATA.PROCESSED_PATH
    trainSet = datasets.ImageFolder(
        os.path.join(proc_path, 'train'), train_transform
    )

    validSet = datasets.ImageFolder(
        os.path.join(proc_path, 'valid'), valid_transform
    )

    BatchSize = cfg.TRAIN.BATCH_SIZE
    NumWorkers = cfg.TRAIN.NUM_WORKERS
    
    trainLoader = DataLoader(
        trainSet, batch_size=BatchSize, shuffle=True, num_workers=NumWorkers
    )

    validLoader = DataLoader(
        trainSet, batch_size=BatchSize, shuffle=False, num_workers=NumWorkers
    )

    dataset_sizes = {
        'train': len(trainSet), 'valid': len(validSet)
    }

    class_names = trainSet.classes
    class_to_idx = trainSet.class_to_idx
    print('\nClass to idx: ', class_to_idx)

    # Use GPU
    isUseGpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if isUseGpu else "cpu")

    # Model summary
    model_ft = models.resnet18(pretrained=True)
    print(); print(model_ft)

    # num_ftrs = model_ft.fc.in_features
    
    # # Here the size of each output sample is set to 2.
    # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    # model_ft.fc = nn.Linear(num_ftrs, 2)

    # model_ft = model_ft.to(device)

    # criterion = nn.CrossEntropyLoss()

    # # Observe that all parameters are being optimized
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


    # model_ft = train_model(
    #     dataloaders, device, dataset_sizes,
    #     model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25,
    # )
    return(0)


if __name__ == '__main__':
    main()


