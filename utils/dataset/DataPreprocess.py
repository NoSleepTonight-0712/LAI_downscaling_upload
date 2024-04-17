# from utils.io.DataReader import getERAFiveVariables, getGCM_PR, getGCM_TA, getERAEachPressuresData
import numpy as np

from utils.io.DataReader import getGCMOriginPack

# def getMeanSigmaValue(variable):
#     if variable == Predictor.PR:
#         data = getGCM_PR()
#     elif variable == Predictor.TA:
#         data = getGCM_TA()


#     data = data.flatten()
#     data = data[~np.isnan(data)]
#     return data.mean(), data.std()

# def getMeanSigmaValueByPressure(predictor, pressure):
#     if predictor < Predictor.Geopotential:
#         data = getERAEachPressuresData(predictor, pressure)
#         data = data.flatten()
#         data = data[~np.isnan(data)]
#         return data.mean(), data.std()
#     elif predictor >= Predictor.Geopotential:
#         data = getERAFiveVariables(VariableERA5Five(predictor, pressure))
#         data = data.flatten()
#         data = data[~np.isnan(data)]
#         return data.mean(), data.std()


# def normalize_data(data, mean_value, std_value) -> np.ndarray:
#     return (data - mean_value) / (std_value)

def getNormalizationParams(variable_index):
    data = getGCMOriginPack()[:, variable_index, :, :]
    return data.mean(), data.std()

def normalize_variable(data, variable_index):
    params = getNormalizationParams(variable_index)
    return (data - params[0]) / (params[1])

def normalize_given_data(data: np.ndarray) -> np.ndarray:
    dataF = data.flatten()
    dataF = dataF[~np.isnan(dataF)]
    return (data - dataF.mean()) / dataF.std()