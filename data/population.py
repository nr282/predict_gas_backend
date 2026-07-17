"""
Provides population data to the calculation. The population data is critical
to monitoring and measuring the amount of Natural Gas Consumption. If the population
increases, then the consumption will increase also.

"""
import datetime
from abc import ABC, abstractmethod
from location import get_list_of_standardizied_name, location

class PopulationData(ABC):
    """
    PopulationData is an abstract class that states the relevant methods, classes
    for population data.

    A major goal is to be able to standardize the population data.
    """

    def __init__(self, population):
        self.df = self.acquire_population_data()
        data_locations = self.df[self.get_location_column()]
        locations_valid, location_not_found = self.check_locations_are_valid(data_locations)

        #Checks to make sure that the locations represented in the population data
        #are indeed acceptable and recognized by the locations module.
        if not locations_valid:
            raise ValueError(f"Invalid locations provided. The location not found is {location_not_found}.")

    def check_locations_are_valid(self, data_locations):
        """
        Check that all locations are considered valid.

        """

        locations_valid = True
        valid_locations = get_list_of_standardizied_name()
        for data_location in data_locations:
            if data_location not in valid_locations:
                locations_valid = False
                return locations_valid, data_location
        return locations_valid, None

    @abstractmethod
    def get_population_for_state_subregion_during_period(self,
                                           state_subregion,
                                           start_datetime: datetime.datetime,
                                           end_datetime: datetime.datetime):
        pass

    @abstractmethod
    def get_population_for_state_subregion(self, state_subregion):
        pass

    @abstractmethod
    def get_population_for_date(self, date: datetime.datetime):
        pass

    @abstractmethod
    def acquire_population_data(self):
        pass

    def get_population_data(self):
        return self.df

    @abstractmethod
    def get_location_column(self) -> str:
        pass

    @abstractmethod
    def get_locations(self) -> list[location]:
        pass