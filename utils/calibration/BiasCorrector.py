from typing import Union
from utils.const.variable_const import Predictand, Predictor, Pressure, VariableV10
from utils.io.DataReader import getFeaturePack
import numpy as np
from skimage.exposure import match_histograms

class BiasCorrector:
    def __init__(self) -> None:
        self.ref_datas = self._initReferenceData() * 100

    def _initReferenceData(self):
        data = getFeaturePack(normalization=False)   # shape like (time, 60, 90)
        return [data[m::12, v, :, :].mean(axis=0) for v in range(9) for m in range(12)]
    
    def value_to_quantile(self, num, fig):
        arr = fig.flatten()
        sorted_arr = np.sort(arr) # Sort the array
        index = np.where(sorted_arr == num)[0][0] # Find index of num
        quantile = index / len(arr)
        return quantile

    def quantile_to_value(self, quantile, fig):
        arr = fig.flatten()
        sorted_arr = np.sort(arr)
        index = int(quantile * len(sorted_arr))
        return sorted_arr[index]
    
    
    def correct(self, data, month, variable: Union[VariableV10, int]):
        if type(variable) == type(1):
            variable_index = variable
        else:
            variable_index = variable.getIndex()

        ref_data = self.ref_datas[12 * variable_index + month]
        return self.calibrate(data, ref_data, variable)

    def calibrate(self, origin_data: np.ndarray, ref_data: np.ndarray, channel) -> np.ndarray:
        # TODO: Choose a calibration method to use
        # return self.calibrate_v1(origin_data, ref_data)
        return self.calibrate_skimage(origin_data, ref_data)
    
    
    def calibrate_skimage(self, origin_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
        return match_histograms(origin_data, ref_data)

    
    def calibrate_v1(self, origin_data: np.ndarray, ref_data: np.ndarray) -> np.ndarray:
        x_shape, y_shape = origin_data.shape
        result_data= np.zeros_like(origin_data)
        for x in range(x_shape):
            for y in range(y_shape):
                q = self.value_to_quantile(origin_data[x, y], origin_data)
                e = self.quantile_to_value(q, ref_data)
                result_data[x, y] = e

        return result_data

