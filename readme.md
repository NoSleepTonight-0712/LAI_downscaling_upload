## A dataset of 0.05-degree leaf area index in China during 1983–2100 based on deep learning network

> The article is available at https://doi.org/10.1038/s41597-024-03948-z

> #### *Article Abstract*  
> 	Leaf Area Index (LAI) is a critical parameter in terrestrial ecosystems, with high spatial resolution data being extensively utilized in various research studies. However, LAI data under future scenarios are typically only available at 1° or coarser spatial resolutions. In this study, we generated a dataset of 0.05° LAI (F0.05D-LAI) from 1983–2100 in a high spatial resolution using the LAI Downscaling Network (LAIDN) model driven by inputs including air temperature, relative humidity, precipitation, and topography data. The dataset spans the historical period (1983–2014) and future scenarios (2015–2100, including SSP-126, SSP-245, SSP-370, and SSP-585) with a monthly interval. It achieves high accuracy (R² = 0.887, RMSE = 0.340) and captures fine spatial details across various climate zones and terrain types, indicating a slightly greening trend under future scenarios. F0.05D-LAI is the first high-resolution LAI dataset and reveals the potential vegetation variation under future scenarios in China, which benefits vegetation studies and model development in earth and environmental sciences across present and future periods.

This dataset is available under Open Science Framework (https://www.doi.org/10.17605/OSF.IO/9QZ4K),containing 5 data records in historical (1983-2014), SSP-126, SSP-245, SSP-370 and SSP-585 (2015-2100). The data is multiplied by 100 and is stored in Int32 data type and GeoTiff file format with WGS 1984 coordinate system. The temporal resolution is 1 month, and the spatial resolution is 0.05 degrees. 

All data were named `lai_<experiment>_<year>_<month>.tif`. The `experiment` field can be one of `historical`, `ssp126`, `ssp245`, `ssp370` and `ssp585`, representing the scenario of the data. The `year` field and `month` field represent the time of the data.

The downscaling procedure is shown in the Figure below:

![Downscaling Procedure](https://media.springernature.com/full/springer-static/image/art%3A10.1038%2Fs41597-024-03948-z/MediaObjects/41597_2024_3948_Fig1_HTML.png?as=webp "Downscaling Procedure")

