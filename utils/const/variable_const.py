import os

class Predictor:
    PR = 1
    TA = 2

    Relative_humidity = 11
    Precipitation_each_pressure = 12
    Temperature_each_pressure = 13

    Geopotential = 21
    Temperature = 22
    Specific_humidity = 23
    Eastward_wind = 24  # ua
    Northward_wind = 25 # va

    V10_Relative_humidity = 31
    V10_Temperature = 32
    V10_precipitation = 33

class Pressure:
    P500 = 'P500'
    P700 = 'P700'
    P850 = 'P850'
    P1000 = 'P1000'

class VariableERA5Five:
    def __init__(self, predictor, pressure) -> None:
        self.predictor = predictor
        self.pressure  = pressure

    
class VariableV10:
    def __init__(self, predictor, pressure) -> None:
        self.predictor = predictor
        self.pressure  = pressure

    def getIndex(self):
        if self.predictor == Predictor.V10_precipitation:
            return 8
        elif self.predictor == Predictor.V10_Relative_humidity:
            if self.pressure == Pressure.P500:
                return 0
            elif self.pressure == Pressure.P700:
                return 1
            elif self.pressure == Pressure.P850:
                return 2
            elif self.pressure == Pressure.P1000:
                return 3
        elif self.predictor == Predictor.V10_Temperature:
            if self.pressure == Pressure.P500:
                return 4
            elif self.pressure == Pressure.P700:
                return 5
            elif self.pressure == Pressure.P850:
                return 6
            elif self.pressure == Pressure.P1000:
                return 7


class Predictand:
    GPP = 11
    NPP = 12
    LAI = 13

    @staticmethod
    def code_to_text(code):
        if code == Predictand.GPP:
            return 'GPP'
        elif code == Predictand.NPP:
            return "NPP"
        elif code == Predictand.LAI:
            return "LAI"
        else:
            return "Unknown"
        
    @staticmethod
    def text_to_code(text: str):
        if text.upper().strip() == 'GPP':
            return Predictand.GPP
        elif text.upper().strip() == 'NPP':
            return Predictand.NPP
        elif text.upper().strip() == 'LAI':
            return Predictand.LAI
        

class FutureExperiment:
    SSP126 = 'ssp126'
    SSP245 = 'ssp245'
    SSP370 = 'ssp370'
    SSP585 = 'ssp585'
    Historical = 'historical'
        