## A Dataset of 0.05-degree LAI in China during 2015–2100 Based on Deep Learning Network

> #### *Article Abstract*  
> Leaf Area Index (LAI) is an important variable for assessing the health of vegetation ecosys-tems. Higher spatial resolution LAI can significantly contribute to the management of eco-systems by providing detailed insights into vegetative cover and dynamics. However, LAI under future scenarios is only available from products with coarse spatial resolutions (mostly 1° or more) in Global Climate Models (GCMs). Here, we generated a dataset of LAI in a high spatial resolution using a LAI Downscaling Network (LAIDN) model. The dataset covers the whole China and spans both historical period (1983-2014) and future scenarios (2015-2100, including SSP-126, SSP-245, SSP-370, and SSP-585) with a spatial resolution of 0.05° and a temporal resolution of one month. The dataset of LAI has a high accuracy (R<sup>2</sup>=0.887, RMSE=0.340) and more spatial details compared to the original LAI from GCMs. The dataset is the first high-resolution dataset of LAI under future scenarios in China, ena-bling vegetation studies across both present and future scenarios. 

This dataset is available under Open Science Framework (https://www.doi.org/10.17605/OSF.IO/9QZ4K),containing 5 data records in historical (1983-2014), SSP-126, SSP-245, SSP-370 and SSP-585 (2015-2100). The data is multiplied by 100 and is stored in Int32 data type and GeoTiff file format with WGS 1984 coordinate system. The temporal resolution is 1 month, and the spatial resolution is 0.05 degrees. 

All data were named `lai_<experiment>_<year>_<month>.tif`. The `experiment` field can be one of `historical`, `ssp126`, `ssp245`, `ssp370` and `ssp585`, representing the scenario of the data. The `year` field and `month` field represent the time of the data.

The downscaling procedure is shown in the Figure below:

![Downscaling Procedure](https://raw.githubusercontent.com/NoSleepTonight-0712/LAI_downscaling_upload/main/.readme_images/techflow.jpg "Downscaling Procedure")

