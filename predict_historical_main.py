# Note that the output data should be furtherly resample from ~0.07 to 0.05 to reach the final result.

from importlib import import_module

import numpy as np

from utils.const.variable_const import FutureExperiment, Predictand
from utils.io.DataReader import getGLASSLAI
from utils.io.PredictionDataReader import getNoChinaMask
from utils.stats.future_stats import getPredictionYear

from backend.predict_historical import predict

# model path
RESUME_MODEL = '/home/zxhy0712/LAI_downscaling_upload/training_result/LAI/M1D1T1_20240201-234535/model_LAI_v1_E1_0.315687_20240201-234604.pytorchmodel'

PREDICTAND_VARIABLE = Predictand.LAI
MODEL_VERSION = 1
DATASET_VERSION = 1


predict(PREDICTAND_VARIABLE, model_version=MODEL_VERSION, 
      dataset_version=DATASET_VERSION, resume_model=RESUME_MODEL, new_data_shape=True)

