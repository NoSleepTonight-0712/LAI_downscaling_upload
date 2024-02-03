import datetime
from loguru import logger
import os
from pathlib import Path
import torch

from utils.const.variable_const import Predictand

class OutputManager():
    def __init__(self, predictand_variable, model_version, dataset_version, training_backend_version) -> None:
        self.predictand_variable = predictand_variable
        self.variable_string = Predictand.code_to_text(predictand_variable)
        self.model_version = model_version
        self.dataset_version = dataset_version
        self.training_backend_version = training_backend_version

        self.start_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

        # init output path
        if not Path(self.getOutputPath()).exists():
            os.makedirs(self.getOutputPath())

        self._init_logger()

        self.max_r2 = 0
        
    def getOutputPath(self):
        return os.path.join('training_result', self.variable_string, f'M{self.model_version}D{self.dataset_version}T{self.training_backend_version}_{self.start_datetime}')

    def _init_logger(self):
        logger.add(os.path.join(self.getOutputPath(), f'log_M{self.model_version}D{self.dataset_version}T{self.training_backend_version}_{self.start_datetime}.log'))
        logger.info(f"Training started at {self.start_datetime}, with model v{self.model_version}, dataset v{self.dataset_version} and training backend v{self.training_backend_version}")

    def writeLog(self, text):
        logger.info(text)

    def writeWarning(self, text):
        logger.warning(text)

    def setModelSummit(self, summit):
        self.max_r2 = summit

    def writeModel(self, info_dict, current_epoch, val_mean_r2):
        timestring = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if val_mean_r2 > self.max_r2:
            self.max_r2 = val_mean_r2
        torch.save(info_dict, os.path.join(self.getOutputPath(), f'model_{self.variable_string}_v{self.model_version}_E{current_epoch}_{val_mean_r2:.6f}_{timestring}.pytorchmodel'))

    def writeModelSummit(self, info_dict, current_epoch, val_mean_r2):
        if val_mean_r2 > self.max_r2:
            self.writeLog(f'Reach new summit at Epoch {current_epoch} with R2 {val_mean_r2}')
            self.writeModel(info_dict, current_epoch, val_mean_r2)


        
            