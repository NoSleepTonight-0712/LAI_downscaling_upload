from unittest import result
from utils.io.DataReader import getGCMCalibratedFuture, getGCMFuture, getPrediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def getGCMFutureYear(variable, experiment, year):
    result = np.zeros((12, 60, 90))
    for m in range(12):
        result[m, :, :] = getGCMFuture(variable, experiment, year, m+1)
    d = result.mean(axis=0)
    return np.where(d < -99, np.nan, d)

def getGCMCalibratedFutureYear(variable, experiment, year):
    result = np.zeros((12, 60, 90))
    for m in range(12):
        result[m, :, :] = getGCMCalibratedFuture(variable, experiment, year, m+1)
    d = result.mean(axis=0)
    return np.where(d < -99, np.nan, d)

def getPredictionYear(variable, experiment, year):
    result = np.zeros((12, 60*16, 90*16))
    # return getPrediction(variable, experiment, year, 5)
    for m in range(12):
        result[m, :, :] = getPrediction(variable, experiment, year, m+1)
    return result.mean(axis=0)

