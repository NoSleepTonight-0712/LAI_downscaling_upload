from typing import Union

from sklearn.linear_model import LinearRegression
from utils.const.variable_const import Predictand, Predictor, Pressure, VariableV10
from utils.io.DataReader import getFeaturePack
import numpy as np
from skimage.exposure import match_histograms

def quantile_mapping(origin_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
    return match_histograms(origin_data, ref_data)

def average_replacement(origin_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
    return origin_data - origin_data.mean(axis=0) + ref_data.mean(axis=0)

def simple_scale(origin_data: np.ndarray, ref_data: np.ndarray):
    return origin_data / origin_data.mean(axis=0) * ref_data.mean(axis=0)

def delta(origin_data, ref_data):
    start_obs_ref = ref_data[:10, :, :].mean(axis=0)

    start_gcm = origin_data[:10, :, :].mean(axis=0)

    increasing = origin_data - start_gcm

    return start_obs_ref + increasing


def delta_scale(origin_data, ref_data):
    start_obs_ref = ref_data[:10, :, :].mean(axis=0)

    start_gcm = origin_data[:10, :, :].mean(axis=0)

    increasing = origin_data / start_gcm

    return start_obs_ref * increasing

def variance_correction(origin_data, ref_data):
    start_obs_ref = ref_data[:10, :, :].mean(axis=0)

    start_gcm = origin_data[:10, :, :].mean(axis=0)

    up = origin_data - start_gcm
    down = start_gcm.std() / start_obs_ref.std()

    return up / down + start_obs_ref

def variance_correction_2(origin_data, ref_data):
    start_obs_ref = ref_data[:10, :, :].mean(axis=0)

    start_gcm = origin_data[:10, :, :].mean(axis=0)

    up = origin_data - start_gcm
    down = start_gcm.std() / start_obs_ref.std()

    return up / down + start_gcm - (start_gcm - start_obs_ref)

def linear_regression(origin_data, ref_data):
    # origin_data = average_replacement(origin_data, ref_data)
    train_size = 20
    result = np.zeros_like(origin_data)
    r2 = []
    for x in range(60):
        for y in range(90):
            Xs = origin_data[:train_size, x, y].reshape((-1, 1))
            Ys = ref_data[:train_size, x, y].reshape((-1, 1))

            regressor = LinearRegression().fit(Xs, Ys)

            r2.append(regressor.score(Xs, Ys))

            result[:, x, y] = regressor.predict(origin_data[:, x, y].reshape((-1, 1))).T

    print(np.mean(r2))
    
    return result