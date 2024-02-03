from importlib import import_module

import numpy as np

from utils.const.variable_const import FutureExperiment, Predictand
from utils.io.DataReader import getGLASSLAI
from utils.io.PredictionDataReader import getNoChinaMask
from utils.stats.future_stats import getPredictionYear

# ! SETTINGS ! 6
PREDICTION_BACKEND_VERSION = 2
RESUME_MODEL = '/home/zxhy0712/LAI_downscaling_upload/training_result/LAI/M1D1T1_20240201-234535/model_LAI_v1_E1_0.315687_20240201-234604.pytorchmodel'

predict = import_module(f'backend.prediction_backend_v{PREDICTION_BACKEND_VERSION}').predict


# ! Start Evaluation
# Extract model params from model path
PREDICTAND_VARIABLE = Predictand.LAI
MODEL_VERSION = 1
DATASET_VERSION = 1


predict(PREDICTAND_VARIABLE, model_version=MODEL_VERSION, 
      dataset_version=DATASET_VERSION, resume_model=RESUME_MODEL, new_data_shape=True)

