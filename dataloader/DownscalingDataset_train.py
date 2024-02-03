# The same with V8, but was refactored to be clearer and use new data shape (60*90)

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.const.variable_const import Predictand
from utils.io.DataReader import (getDEM, getFeaturePack,
                                 getGLASSGPP, getGLASSLAI, getGLASSNPP)


class DownscalingDataset(Dataset):
    def __init__(self, data_dir: str, variable, device='cpu', normalization=True) -> None:
        super().__init__()
        self.data_dir: str = data_dir
        self.variable: str = variable
        self.device = device

        self.datapack = getFeaturePack(normalization)

    def _getRefData(self, index_or_year, none_or_month=None):
        """get Reference data. NoData will be filled with 0."""
        if self.variable == Predictand.GPP:
            d = getGLASSGPP(index_or_year, none_or_month, new_data_shape=True)
        elif self.variable == Predictand.NPP:
            d = getGLASSNPP(index_or_year, none_or_month, new_data_shape=True)
        elif self.variable == Predictand.LAI:
            d = getGLASSLAI(index_or_year, none_or_month, new_data_shape=True)
            
        return np.where(np.isnan(d), 0, d)
           
    def __len__(self):
        return self.datapack.shape[0]
    
    def __getitem__(self, idx):
        predictors = torch.tensor(self.datapack[idx, :, :, :], dtype=torch.float32)

        DEM = getDEM()

        label = torch.tensor(self._getRefData(idx))

        return idx, [predictors, DEM], label 

