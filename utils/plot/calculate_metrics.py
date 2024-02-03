import numpy as np
import torch
import torch.nn as nn
from torchmetrics.functional import r2_score as r2_score_torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

device = 'cuda'

MSE = nn.MSELoss().to(device)
MAE = nn.L1Loss().to(device)

def getMetrics(origin_output, origin_label, norm=False):
    out_f = origin_output.flatten()
    out_f[out_f < 0] = 0

    ref_f = origin_label.flatten()
    nonan_mask = (np.isnan(out_f)) | (np.isnan(ref_f))
    out_nonan = out_f[~nonan_mask]
    ref_nonan = ref_f[~nonan_mask]

    rmse = np.sqrt(mean_squared_error(out_nonan, ref_nonan))
    mae = mean_absolute_error(out_nonan, ref_nonan)
    r2   = r2_score(out_nonan, ref_nonan)

    return rmse, mae, r2, out_nonan, ref_nonan

def getMetricsTorch(origin_output, origin_label, norm=False):
    out_f = origin_output.flatten()
    out_f[out_f < 0] = 0

    ref_f = origin_label.flatten()
    nonan_mask = (torch.isnan(out_f)) | (torch.isnan(ref_f))
    out_nonan = out_f[~nonan_mask]
    ref_nonan = ref_f[~nonan_mask]

    rmse = torch.sqrt(MSE(out_nonan, ref_nonan)).detach().cpu().item()
    mae = MAE(out_nonan, ref_nonan).detach().cpu().item()
    r2   = r2_score_torch(out_nonan, ref_nonan).detach().cpu().item()

    return rmse, mae, r2, out_nonan, ref_nonan

def getNoNanDataTorch(origin_output, origin_label):
    out_f = origin_output.flatten()
    out_f[out_f < 0] = 0

    ref_f = origin_label.flatten()
    nonan_mask = (torch.isnan(out_f)) | (torch.isnan(ref_f))
    out_nonan = out_f[~nonan_mask]
    ref_nonan = ref_f[~nonan_mask]

    return out_nonan, ref_nonan

def calcWeightTorch(out_nonan, ref_nonan):
    pass
