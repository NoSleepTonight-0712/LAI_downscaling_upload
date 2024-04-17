# Please implement this code to start training.

import cv2
import torch
import os
import numpy as np
from osgeo import gdal_array

from utils.const.variable_const import FutureExperiment, Predictand, Predictor, Pressure, VariableERA5Five
from utils.datetime.datetime import index_to_yearMonth

from typing import Union

from utils.const.global_const import GCM_SHAPE, TARGET_SHAPE, TARGET_SHAPE_CV, OLD_TARGET_SHAPE_CV, CalibrationType
FutureExperiment


def getNormalizationParams(variable_index):
    if variable_index == 8:
        scale = 2592
    else:
        scale = 1
    data = getGCMOriginPack(new_shape=True)[:, variable_index, :, :] * scale
    return data.mean(), data.std()

def normalize_variable(data, variable_index):
    params = getNormalizationParams(variable_index)
    return (data - params[0]) / (params[1])

def normalize_given_data(data: np.ndarray) -> np.ndarray:
    dataF = data.flatten()
    dataF = dataF[~np.isnan(dataF)]
    return (data - dataF.mean()) / dataF.std()

def _resize_GLASS(origin_label):
    return cv2.resize(cv2.resize(origin_label, TARGET_SHAPE_CV, 
                                    interpolation=cv2.INTER_NEAREST), TARGET_SHAPE_CV, 
                                    interpolation=cv2.INTER_CUBIC)

def getDEM() -> torch.Tensor:
    # This function should return a DEM
    return torch.tensor(np.load('/home/zxhy0712/CMIP_PP/data/DEM/DEM_960_1440.npy'), dtype=torch.float32).unsqueeze(0)

def _getGLASS_LAI_by_yearmonth(year, month) -> np.ndarray:
    # Load the observation LAI in high resolution (GLASS)
    return gdal_array.LoadFile(f'/home/zxhy0712/CMIP_PP/data/GLASS/LAI/{year}/{year}_{month}.tif')

def _getGLASS_LAI_by_index(index) -> np.ndarray:
    return _getGLASS_LAI_by_yearmonth(*index_to_yearMonth(index))

def getGLASSLAI(index_or_year, none_or_month=None) -> np.ndarray:
    if none_or_month == None:
        d = _getGLASS_LAI_by_index(index_or_year)
    else:
        d = _getGLASS_LAI_by_yearmonth(index_or_year, none_or_month)

    return _resize_GLASS(d)


def getERA_PR(normalization=False) -> np.ndarray:
    if not normalization:
        # return ERA precipitation data
        return np.load('/home/zxhy0712/CMIP_PP/data/pr/pr.npy')
    else:
        return normalize_given_data(getERA_PR(normalization=False))

def getGLASS_NODATA(stack=1) -> np.ndarray:
    if stack == 1:
        return np.isnan(getGLASSLAI(0))
    else:
        d = getGLASS_NODATA(stack=1)
        return np.expand_dims(np.array([d] * stack), 1)
    
def getERAEachPressuresData(predictor, pressure, normalization=False):
    if predictor == Predictor.Relative_humidity:
        variable_string = 'r'
    elif predictor == Predictor.Temperature_each_pressure:
        variable_string = 't'

    if not normalization:
        # return ERA 5 data of Relative_humidity and temperature in each pressure
        return np.load(f'/home/zxhy0712/CMIP_PP/data/ERA_Pressures/{variable_string}/{pressure}/{variable_string}_{pressure}.nc.npy').astype(np.float32)
    else:
        return normalize_given_data(getERAEachPressuresData(predictor, pressure, normalization=False))

def getFeaturePack(normalization=True) -> np.ndarray:
    era_datapack = np.array([
            getERAEachPressuresData(v, p, normalization=normalization) 
            for v in [Predictor.Relative_humidity, Predictor.Temperature_each_pressure] 
            for p in [Pressure.P500, Pressure.P700, Pressure.P850, Pressure.P1000]
            ])
    era_datapack = np.swapaxes(era_datapack, 0, 1)
    PR = getERA_PR(normalization=normalization)
    PR = np.array([cv2.resize(PR[idx, :, :], GCM_SHAPE[::-1], interpolation=cv2.INTER_CUBIC) for idx in range(PR.shape[0])])
    PR = np.expand_dims(PR, 1)
    return np.concatenate([era_datapack, PR], axis=1)

# Read GCM
def getGCM_PR() -> torch.Tensor:
    # read the precipitation from GCM
    data = np.load('/home/zxhy0712/CMIP_PP/data/pr_GCM/pr.npy')
    result = np.zeros((384, 60, 90))
    for t in range(384):
        result[t, :, :] = cv2.resize(data[t, :, :], (90, 60), interpolation=cv2.INTER_CUBIC)
    return result

def getGCM_TA() -> torch.Tensor:
    return np.load('/home/zxhy0712/CMIP_PP/data/ta_GCM/ta.npy')

def getGCMOriginPack() -> np.ndarray:
    """return shape: (384, 9, 60, 90)"""

    # this is a GCM data pack as following shape: (384, 9, 60, 90)
    # axis 1 (384) is the time from 1984.1 to 2014.12
    # axis 2 (9) is the variables, following [r500, r700, r850, r1000, t500, t700, t850, t1000, pr].
    # r = relative humidity, t = air temperature, pr = precipitation
    # axis 3 (60) and axis 4 (90) are the size of GCM data.
    d = np.load('/home/zxhy0712/CMIP_PP/data/GCM_pack/GCM_pack.npy')
    for v in range(9):
        d[:, v, :, :] = normalize_variable(d[:, v, :, :], v)
    return d


def getGCMCalibratedPack(calibration_type=CalibrationType.Quantile) -> np.ndarray:
    if calibration_type == CalibrationType.Quantile:
        return np.load('/home/zxhy0712/CMIP_PP/data/GCM_V10_variables_calibrated/GCM_calibrated.npy')
    elif calibration_type == CalibrationType.Replacement:
        return np.load('/home/zxhy0712/CMIP_PP/data/GCM_V10_variables_calibrated/GCM_calibrated_replace.npy')
    elif calibration_type == CalibrationType.Hybrid_Quantile_Replacement:
        return np.load('/home/zxhy0712/CMIP_PP/data/GCM_V10_variables_calibrated/GCM_calibrated_hybrid_quantile+replacement.npy')
    elif calibration_type == CalibrationType.Hybrid_Replacement_Quantile:
        return np.load('/home/zxhy0712/CMIP_PP/data/GCM_V10_variables_calibrated/GCM_calibrated_hybrid_replacement+quantile.npy')
    elif calibration_type == CalibrationType.Latest:
        return np.load('/home/zxhy0712/CMIP_PP/data/GCM_V10_variables_calibrated/GCM_calibrated.npy')

def getGCMCalibratedPackNormalized(calibration_type=CalibrationType.Quantile) -> np.ndarray:
    d = getGCMCalibratedPack(calibration_type)
    for v in range(9):
        d[:, v, :, :] = normalize_given_data(d[:, v, :, :])

    return d

def getFutureDatapack(experiment, normalization=False) -> np.ndarray:
    if not normalization:
        # this is a future feature pack as following shape: (1032, 9, 60, 90)
        # Note the future feature pack is from GCMs.
        # axis 1 (1032) is the time from 2015.1 to 2100.12
        # axis 2 (9) is the variables, following [r500, r700, r850, r1000, t500, t700, t850, t1000, pr].
        # r = relative humidity, t = air temperature, pr = precipitation
        # axis 3 (60) and axis 4 (90) are the size of GCM data.
        
        return np.load(f'/home/zxhy0712/CMIP_PP/data/Future_datapack/future_datapack_{experiment.strip().lower()}.npy')
    else:
        d = getFutureDatapack(experiment, normalization=False)
        for v in range(9):
            d[:, v, :, :] = normalize_variable(d[:, v, :, :], v)
        return d

def getFutureDatapackCalibrated(experiment, normalization=True, used_data_version=1) -> np.ndarray:
    if not normalization:
        if used_data_version == 1:
            return np.load(f'/home/zxhy0712/CMIP_PP/data/Future_calibrated_datapack/future_calibrated_{experiment.strip().lower()}.npy')
        else:
            return np.load(f'/home/zxhy0712/CMIP_PP/data/Future_calibrated_datapack/future_calibrated_{experiment.strip().lower()}_v{used_data_version}.npy')
    else:
        d = getFutureDatapackCalibrated(experiment, normalization=False, used_data_version=used_data_version)
        for v in range(9):
            d[:, v, :, :] = normalize_variable(d[:, v, :, :], v)
        return d
    
