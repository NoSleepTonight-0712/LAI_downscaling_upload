# Set learning rate manually.
# Set learning rate by range.

import datetime
import os
import numpy as np

from loguru import logger

from utils.const.variable_const import Predictand
from utils.io.DataExportManager import DataExportManager
from utils.stats.log_stats import getQuantile, writeLog

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import importlib

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import r2_score as r2_score_torch

from utils.const.global_const import DEVICE
from utils.dataset.DatasetSplit import splitDatasetByMonth
from utils.io.DataReader import getGLASS_NODATA

MSE = nn.MSELoss().to(DEVICE)
MAE = nn.L1Loss().to(DEVICE)

# SETTINGS
TRAIN_PROPORTION = 0.7
VAL_PROPORTION = 0.15


# Train Core
def predict(target_variable, model_version, dataset_version, experiment, resume_model, new_data_shape=True, prediction_version=1):
    dataExportManager = DataExportManager(target_variable, experiment, prediction_version)

    isResume = resume_model != None
    timestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    logger.info(f"Prediction started at {timestring}, with model v{model_version}, dataset v{dataset_version}, model file {resume_model}")

    NeuralNetwork = importlib.import_module(f'model.model_PP_v{model_version}').NeuralNetwork
    DownscalingDataset = importlib.import_module(f'dataloader.DownscalingDataset_PP_v{dataset_version}').DownscalingDataset
    
    model: Module = NeuralNetwork().to(DEVICE)
    dataset: Dataset = DownscalingDataset('data', target_variable, device=DEVICE, experiment=experiment, used_data_version=prediction_version)
    
    logger.info(f'Dataset length: {len(dataset)}')
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.enabled = True

    dataloader  = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=False, persistent_workers=True)
    
    nodata_stack1 = torch.tensor(getGLASS_NODATA(new_data_shape=new_data_shape))

    def setDevice(feature):
        feature = feature[0].to(DEVICE), feature[1].to(DEVICE)
        return feature
    
    # extract from resume
    if isResume:
        checkpoint = torch.load(resume_model)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    model.eval()

    logger.info('Start evaluating each sample in test set.')


    # Val Loop
    for idx, feature in dataloader:
        feature = setDevice(feature)
        output  = model(feature).squeeze().detach().cpu()
        feature[0].detach()
        feature[1].detach()

        # afterprocess
        output = torch.where(nodata_stack1, torch.nan, output)
        output = torch.where(output < 0, 0, output)
        output = output.detach().cpu().numpy()

        logger.info(f'Sample Index {idx[0]} Mean={np.nanmean(output)}')
        
        output = np.where(np.isnan(output), 1 << 31 - 1, output)
        dataExportManager.write(idx[0], output)


