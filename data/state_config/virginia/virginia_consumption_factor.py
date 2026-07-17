"""
Virginia module aims to calculate core quantities that can be used in other modules.

Virginia will look to take in population estimates.

"""
import sys
import os
import pandas as pd
import data.state_config.state_config as state_config
import data.population as population
import datetime
from location import location
from typing import List
import numpy as np
from scipy.interpolate import interp1d
from utils import *

class VirginiaConsumptionFactorCalculation(state_config.ConsumptionConfiguration):
    """
    Calculates the consumption factor for Virginia.
    """

    def __init__(self, population, weather, wind):
        """
        Calculates the consumption factor for Virginia.

        """

        super().__init__(population, weather, wind)


    def get_population(self, state_subregion):

        return


    def get_weather_temperature(self, state_subregion):
        pass


    def get_wind(self, state_subregion):
        pass


    def calculate_consumption_factor(self, county):
        pass


    def get_all_state_subregions(self):
        pass


    def get_weights_for_state_subregions(self):
        pass

    def get_state_name(self):
        pass

    def calculate_consumption_factor_for_state(self):
        pass

class VirginiaPopulationData(population.PopulationData):
    """
    Virginia Population Data placed into the PopulationData interface.

    """

    def __init__(self):
        super().__init__(self)

    def acquire_population_data(self):
        replacement_column_name = "County/State"
        population_data = pd.read_csv(os.path.join(get_base_path(), "data", "population", "virginia_population.csv"))
        population_data.rename(columns={"Unnamed: 0": replacement_column_name}, inplace=True)
        population_data[replacement_column_name] = population_data[replacement_column_name].apply(
            lambda x: x[1:] if x[0] == "." else x)

        return population_data

    def get_population_for_state_subregion(self, state_subregion):
        pass

    def get_population_for_date(self, date: datetime.datetime):
        pass

    def get_population_data(self):
        return self.df

    def get_location_column(self):
        return "County/State"

    def get_locations_by_name(self) -> List[str]:
        locations = list(self.df[self.get_location_column()].unique())
        return locations

    def get_locations(self) -> List[location]:
        locations = []
        for index, row in self.df.iterrows():
            locations.append((row["longtitude"], -1 * abs(row["lattitude"])))
        return locations

    def get_population_for_state_subregion_during_period(self,
                                                       location: location,
                                                       start_datetime: datetime.datetime,
                                                       end_datetime: datetime.datetime):
        """
        Provides population data back for every date between (1) start_datetime and (2) end_datetime.
        """

        loc_vals = self.df[self.df["longtitude"].apply(lambda x: (abs(x - location[0]) < 0.002))]
        loc_vals = loc_vals[loc_vals["lattitude"].apply(lambda x: (abs(x - (-1 * location[1])) < 0.002))]

        if len(loc_vals) == 1:
            if all([year_str in loc_vals for year_str in ["2019",
                                                          "2020",
                                                          "2021",
                                                          "2022",
                                                          "2023",
                                                          "2024"]]):


                vals = {"2019": float(loc_vals["2019"].iloc[0].replace(",", "")),
                        "2020": float(loc_vals["2020"].iloc[0].replace(",", "")),
                        "2021": float(loc_vals["2021"].iloc[0].replace(",", "")),
                        "2022": float(loc_vals["2022"].iloc[0].replace(",", "")),
                        "2023": float(loc_vals["2023"].iloc[0].replace(",", "")),
                        "2024": float(loc_vals["2024"].iloc[0].replace(",", ""))
                        }

                s = pd.Series(vals)
                df = pd.DataFrame(s)
                df["Day_Count"] = np.array([0, 365, 731, 1096, 1461, 1826])
                df.rename(columns={0: "Population"}, inplace=True)
                f_cubic = interp1d(df["Day_Count"].values, df["Population"].values, kind='cubic')
                dt = pd.date_range(start=start_datetime, end=end_datetime, freq="D")
                df = pd.DataFrame(dt)
                df["Day_Count"] = df.index
                df["Daily_Population"] = df["Day_Count"].apply(lambda x: f_cubic(x))
                df = df.set_index(0).drop(columns=["Day_Count"])
                df.index.name = "Date"
                df.reset_index(inplace=True)
                return df
        else:
            raise ValueError("Multiple locations found.")

if __name__ == "__main__":
    pass




