"""
Module aims to handle forecast weather. There is a strong distinction between forecast weather and
historical weather. Historical Weather is often seen as being known perfectly. Forecast Weather is viewed
as not being known perfectly.

The module will setup up the Forecast Weather. The Forecast Weather will be different than
Historical Weather data.

I will need to calculate the difference between:
    + the variance between (1) historical weather and (2) forecast weather
    + the variance provided by the weather provider.


"""


#TODO: The goal will require us to make the code abstract.
from abc import ABC, abstractmethod

class ForecastWeather(ABC):
    """
    Forecast Weather will require (1) variance between historical weather and forecast weather.

    """

    @abstractmethod
    def acquire_forecast_weather(self):
        pass


class PrescientWeather(ForecastWeather):
    """
    Prescient Weather used here will be in a Forecast setting.

    """

    def __init__(self):
        pass

    #TODO: Add methods to the object. What will I do with the prescient weather.
    #TODO: Once we have developed this weather data.
    #TODO: We can then pipe this into the statistical model.
    #TODO:


    def acquire_forecast_weather(self):
        pass









