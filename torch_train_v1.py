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
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
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
            if phase == 'valid' and epoch_acc > best_acc:
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

    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225]
    )

    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    }

    ClassNames = list(data_transforms.keys())
    print('\nClass names: ', ClassNames)

    # Datasets
    ProcPath = cfg.DATA.PROCESSED_PATH

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(ProcPath, x), data_transforms[x])
        for x in ClassNames
    }

    trainSet = image_datasets['train']

    class_indexes = trainSet.class_to_idx
    print('class to index: ', class_indexes)

    trainX, trainY = trainSet[0]
    print('\nimage:')
    print(trainX)
    print('label:', trainY)    

    BatchSize = cfg.TRAIN.BATCH_SIZE
    NumWorkers = cfg.TRAIN.NUM_WORKERS
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'],
            batch_size=BatchSize, shuffle=True,  num_workers=NumWorkers
        ),
        'valid': DataLoader(image_datasets['valid'],
            batch_size=BatchSize, shuffle=False, num_workers=NumWorkers
        )
    }

    dataset_sizes = {
        x: len(image_datasets[x]) for x in ClassNames
    }
    print('dataset sizes: ', dataset_sizes)

    # Use GPU
    isUseGpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if isUseGpu else "cpu")

    # Model summary
    model_ft = models.resnet18(pretrained=True)
    # print(); print(model_ft)

    num_ftrs = model_ft.fc.in_features
    
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=momentum)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    # Train and evaluate
    print()
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    model_ft = train_model(
        dataloaders, device, dataset_sizes,
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=num_epochs,
    )
    return(0)


if __name__ == '__main__':
    main()


