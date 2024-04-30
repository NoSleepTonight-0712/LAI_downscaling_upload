## A Dataset of 0.05-degree LAI in China during 2015–2100 Based on Deep Learning Network

> #### *Article Abstract*  
> 	Higher spatial resolution Leaf Area Index (LAI) can significantly contribute to forest management, and vegetation dynamics detection, and can be used as an input for land surface models. However, LAI under future scenarios is only available from products with coarse spatial resolutions (typically 1° or more) in Global Climate Models (GCMs). In this study, we generated a dataset of LAI in a high spatial resolution using the LAI Downscaling Network (LAIDN) model which receives air temperature, relative humidity, precipitation, and topography data as input. The dataset spans the historical period (1983-2014) and future scenarios (2015-2100, including SSP-126, SSP-245, SSP-370, and SSP-585) with a spatial resolution of 0.05° at monthly intervals. It has high accuracy (R<sup>2</sup>=0.887, RMSE=0.340) and more spatial details compared to the original LAI from GCMs, revealing a more significant greening trend in vegetation under high emission scenarios (SSP-370, SSP-585). It is the first high-resolution LAI dataset under future scenarios in China, which benefits vegetation studies and model development in earth and environmental sciences across present and future periods.

This dataset is available under Open Science Framework (https://www.doi.org/10.17605/OSF.IO/9QZ4K),containing 5 data records in historical (1983-2014), SSP-126, SSP-245, SSP-370 and SSP-585 (2015-2100). The data is multiplied by 100 and is stored in Int32 data type and GeoTiff file format with WGS 1984 coordinate system. The temporal resolution is 1 month, and the spatial resolution is 0.05 degrees. 

All data were named `lai_<experiment>_<year>_<month>.tif`. The `experiment` field can be one of `historical`, `ssp126`, `ssp245`, `ssp370` and `ssp585`, representing the scenario of the data. The `year` field and `month` field represent the time of the data.

The downscaling procedure is shown in the Figure below:

![Downscaling Procedure](https://raw.githubusercontent.com/NoSleepTonight-0712/LAI_downscaling_upload/main/.readme_images/techflow.jpg "Downscaling Procedure")

