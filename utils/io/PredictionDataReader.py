from osgeo import gdal_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from utils.const.variable_const import Predictand, FutureExperiment
from utils.io.DataReader import getGLASS_NODATA

def yearmonth_to_index(year, month):
    return (year - 2015) * 12 + month - 1


def getPrediction(predictand, experiment, year_or_index, month_or_none, data_version=2):
    # data_version = 2 means replacement calibration's result
    if month_or_none == None:
        index = year_or_index
    else:
        index = yearmonth_to_index(year_or_index, month_or_none)

    predictand = Predictand.code_to_text(predictand)

    filelocation = f'{predictand}_E{experiment}_V{data_version}'
    filename = f'{predictand}_{index}.tif'

    return gdal_array.LoadFile(os.path.join('data_export', filelocation, filename))

def getNoChinaMask():
    nochina_mask = gdal_array.LoadFile('data/info/Climage_region.tif')
    nochina_mask = cv2.resize(nochina_mask, (1440, 960), interpolation=cv2.INTER_NEAREST)
    nodata = getGLASS_NODATA(new_data_shape=True)
    return (~nodata) & (nochina_mask == 0)
