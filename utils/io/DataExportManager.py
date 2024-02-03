from osgeo import gdal
import os
from pathlib import Path

from utils.const.variable_const import Predictand

from loguru import logger

class DataExportManager:
    def __init__(self, variable, experiment, version) -> None:
        self.variable = Predictand.code_to_text(variable)
        self.spatial_ref = gdal.Open('/home/zxhy0712/CMIP_PP/data/info/Ref_1440x960.tif')
        self.experiment = experiment
        self.version = version

        self._init()

    def _init(self):
        self.dest = os.path.join(f'/home/zxhy0712/CMIP_PP/data_export/{self.variable.lower()}/{self.experiment.lower()}')
        if not Path(self.dest).exists():
            os.makedirs(self.dest)

        logger.add(os.path.join(self.dest, f'{self.variable}_E{self.experiment}_V{self.version}.log'))

    def write(self, idx, img):
        filename = f'{self.variable}_{self.experiment}_{idx}.tif'.lower()
        self._writeTiff(img, os.path.join(self.dest, filename))
    

    def _writeTiff(self, array, file, dtype=gdal.GDT_Int32, nodata=(1 << 30)):
        rows = array.shape[0]
        columns = array.shape[1]
        isMultiBands = False
        if len(array.shape) == 3:
            # 多波段文件
            isMultiBands = True
            dim = array.shape[2]
        else:
            dim = 1

        driver: gdal.Driver = gdal.GetDriverByName('GTiff')

        dst_ds: gdal.Dataset = driver.Create(file, columns, rows, dim, dtype, options=['COMPRESS=LZW'])
        
        dst_ds.SetGeoTransform(self.spatial_ref.GetGeoTransform())
        dst_ds.SetProjection(self.spatial_ref.GetProjection())

        for i in range(dim):
            if isMultiBands:
                b = dst_ds.GetRasterBand(i+1)
                b.SetNoDataValue(nodata)
                b.WriteArray(array[:, :, i])
            else:
                b = dst_ds.GetRasterBand(i+1)
                b.SetNoDataValue(nodata)
                b.WriteArray(array[:, :])