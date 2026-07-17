"""
State Config aims to provide an abstract method.


"""

from abc import ABC, abstractmethod
import numpy as np

class ConsumptionConfiguration(ABC):
    """
    Consumption Configuration states the interface that must be implemented
    by each state to get a population weather-driven measure that should be
    correlated to natural gas consumption.

    Generally, this module must calculate information from a number of sources including:
        1. population
        2. weather temperature
        3. wind

    """

    def __init__(self, population, weather, wind):
        pass

    @abstractmethod
    def get_population(self, state_subregion):
        pass

    @abstractmethod
    def get_weather_temperature(self, state_subregion):
        pass

    @abstractmethod
    def get_wind(self, state_subregion):
        pass

    @abstractmethod
    def calculate_consumption_factor(self, county):
        pass

    @abstractmethod
    def get_all_state_subregions(self):
        pass

    @abstractmethod
    def get_weights_for_state_subregions(self):
        pass

    @abstractmethod
    def get_state_name(self):
        pass

    def calculate_consumption_factor_for_state(self):
        pass








