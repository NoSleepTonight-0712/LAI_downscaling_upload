# Set learning rate manually.
# Set learning rate by range.

import datetime
import os

from utils.io.OutputManager import OutputManager
from utils.stats.log_stats import writeLog

# from utils.stats.calculate_metrics import getMetricsTorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics import R2Score
from torchmetrics.functional import r2_score as r2_score_torch

from utils.const.global_const import DEVICE
from utils.dataset.DatasetSplit import splitDatasetByMonth
from utils.io.DataReader import getGLASS_NODATA

from model.model import NeuralNetwork
from dataloader.DownscalingDataset_train import DownscalingDataset

MSE = nn.MSELoss().to(DEVICE)
MAE = nn.L1Loss().to(DEVICE)

# SETTINGS
TRAIN_PROPORTION = 0.7
VAL_PROPORTION = 0.15

BATCH_SIZE = 2
MAX_EPOCHES = 5000

# Learning Rate Function
def GET_LEARNING_RATE(epoch):
    if epoch < 8:
        return 0.01
    elif epoch < 110:
        return 0.001
    elif epoch < 150:
        return 0.01
    elif epoch < 400:
        return 0.001
    elif epoch < 600:
        return 1e-6
    else:
        return 1e-7


# Train Core
def train(target_variable):

    # These three variables were useful in debug, but abandoned now.
    model_version = 1
    dataset_version = 1
    training_backend_version = 1


    model: Module = NeuralNetwork().to(DEVICE)
    dataset: Dataset = DownscalingDataset('data', target_variable, DEVICE)

    outputManager = OutputManager(target_variable, model_version, dataset_version, training_backend_version)

    train_size = int(TRAIN_PROPORTION * len(dataset))
    val_size = int(VAL_PROPORTION * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = \
        splitDatasetByMonth(dataset, train_count=train_size, test_count=test_size, val_count=val_size)
    
    outputManager.writeLog(f'Datset length: Train {len(train_set)} : Val {len(val_set)} : Test {len(test_set)}')
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.enabled = True

    train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4, shuffle=True, drop_last=True, persistent_workers=True)
    val_dataloader  = DataLoader(val_set, batch_size=1, num_workers=4, shuffle=True, persistent_workers=True)
    
    criterion = nn.MSELoss().to(DEVICE)

    optimizer = optim.Adam(params=model.parameters())
    
    nodata = torch.tensor(getGLASS_NODATA(stack=BATCH_SIZE))[:, :, 20:-20, 20:-20].flatten()
    nodata_stack1 = torch.tensor(getGLASS_NODATA())[20:-20, 20:-20].flatten()

    def setDevice(feature, label):
        feature = feature[0].to(DEVICE), feature[1].to(DEVICE)
        label = label.to(DEVICE)
        return feature, label
    

    for e in range(1, MAX_EPOCHES + 1):
        # ! Set Learning Rate
        for pg in optimizer.param_groups:
            pg['lr'] = GET_LEARNING_RATE(e)
        
        train_loss_record = []
        test_loss_record = []
        train_r2_record = []
        test_r2_record = []
        model.train()

        outputManager.writeLog(f"current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        # Train Loop
        for idx, feature, label in train_dataloader:
            feature, label = setDevice(feature, label)
            output = model(feature)
            output = output.squeeze()[:, 20:-20, 20:-20].flatten()

            optimizer.zero_grad()

            # afterprocess
            out_maskvoid = output[~nodata]
            out_maskvoid[out_maskvoid < 0] = 0

            label = label[:, 20:-20, 20:-20].flatten()[~nodata]
            label[label < 0] = 0

            loss = criterion(out_maskvoid, label)
            loss.backward()

            optimizer.step()

            r2 = r2_score_torch(out_maskvoid, label)

            outputManager.writeLog('Epoch: %d/%d, Loss: %.2f, R2: %.4f' % (e, MAX_EPOCHES, loss.item(), r2.item()))
            train_loss_record.append(loss.item())
            train_r2_record.append(r2.item())

        # Val Loop
        model.eval()
        for idx, feature, label in val_dataloader:
            feature, label = setDevice(feature, label)
            output  = model(feature).squeeze().detach()[20:-20, 20:-20]
            label   = label.squeeze().detach()[20:-20, 20:-20]
            feature[0].detach()
            feature[1].detach()

            # afterprocess
            out_maskvoid = output.flatten()[~nodata_stack1]
            out_maskvoid[out_maskvoid < 0] = 0

            label = label.flatten()[~nodata_stack1]
            label[label < 0] = 0

            rmse = torch.sqrt(MSE(out_maskvoid, label)).detach().cpu().item()
            mae = MAE(out_maskvoid, label).detach().cpu().item()
            r2   = r2_score_torch(out_maskvoid, label).detach().cpu().item()

            test_loss_record.append(rmse)
            test_r2_record.append(r2)

        writeLog(outputManager, e, train_loss_record, train_r2_record, test_loss_record, test_r2_record)

        # Write model
        if ((sum(test_r2_record) / len(test_r2_record)) > 0.3) and e in [10, 25, 50, 70] or (e > 70 and e % 15 == 0) or (e > 200 and e % 10 == 0):
            outputManager.writeModel({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, e, sum(test_r2_record) / len(test_r2_record))

        # write model when reach a new summit
        outputManager.writeModelSummit({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, e, sum(test_r2_record) / len(test_r2_record))
