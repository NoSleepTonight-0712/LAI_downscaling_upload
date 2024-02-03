from importlib import import_module

from utils.const.variable_const import FutureExperiment, Predictand

# ! SETTINGS ! 6
ALL_RESUME_MODEL = ['training_result/LAI/M9D10T4_20230826-104853/model_LAI_v9_E437_0.901456_20230826-114546.pytorchmodel']
for RESUME_MODEL in ALL_RESUME_MODEL:
      for EXPERIMENT in [FutureExperiment.SSP370, FutureExperiment.SSP585, FutureExperiment.SSP126, FutureExperiment.SSP245]:
            PREDICTION_BACKEND_VERSION = 1
            # EXPERIMENT = FutureExperiment.SSP370

            predict = import_module(f'backend.prediction_backend_v{PREDICTION_BACKEND_VERSION}').predict
            PREDICTION_VERSION = 2


            # ! Start Evaluation
            # Extract model params from model path
            PREDICTAND_VARIABLE = Predictand.text_to_code(RESUME_MODEL.split('/')[1])
            MODEL_VERSION = int(RESUME_MODEL.split('/')[2].split('_')[0].split('D')[0].replace('M', ''))
            DATASET_VERSION = int(RESUME_MODEL.split('/')[2].split('_')[0].split('D')[1].split('T')[0])
            DATASET_VERSION = 12


            predict(PREDICTAND_VARIABLE, model_version=MODEL_VERSION, 
                  dataset_version=DATASET_VERSION, resume_model=RESUME_MODEL, new_data_shape=True, experiment=EXPERIMENT, prediction_version=PREDICTION_VERSION)
            
