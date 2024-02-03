from utils.const.variable_const import Predictand

from backend.training_backend import train

PREDICTAND_VARIABLE = Predictand.LAI

train(PREDICTAND_VARIABLE)
