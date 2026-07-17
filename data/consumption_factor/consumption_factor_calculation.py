"""
The following module aims to calculate the consumption factor. The consumption factor
is a calculation that takes in a number of factors and develops a measure of how much
natural gas consumption there is. Factors that it should consider are:
    1. Population
    2. Economy
    3. Weather
    4. Wind

Demand-Side Factors:
------------------------------------------------------------------------------------------------------------------------

    Weather:
        Colder winters increase demand for heating,
        while hotter summers increase demand for electricity,
        as natural gas-fired power plants are used to meet cooling loads.

    Economic Growth:
        A stronger economy generally leads to higher consumption,
        particularly in the industrial and commercial sectors,
        which use gas for manufacturing and other processes.

    Data Centers:
        The rapid growth of power-hungry data centers in Virginia
        is driving immense increases in energy demand, directly contributing to
        higher natural gas consumption to generate electricity.

        Data Center map provided here is interesting:
            1. https://www.datacentermap.com/research/

    Appliance Efficiency & Lifestyle:
        Residential use is affected by the age and efficiency of gas appliances and
        changes in living habits, such as increased use of gas fireplaces, stoves,
        and water heaters.

Supply-Side & Other Factors:
------------------------------------------------------------------------------------------------------------------------

    Availability of Other Fuels:
        The availability and prices of alternative energy sources,
        such as oil and renewables, can influence natural gas consumption,
        as they may be used as substitutes, particularly in power generation.

    Infrastructure:
        The extent of natural gas consumption infrastructure in buildings
        and its use in natural gas-fired power plants directly influences consumption levels.

    Global and Market Conditions:
        External factors like the wholesale market for natural gas and geopolitical issues can create price volatility,
        affecting both consumption and the cost to consumers

The census.gov seems to have good information. The information is provided here:
    1. https://www.census.gov/data/developers/data-sets.html#accordion-a8ec3eb6f0-item-e94424333b

"""

from data.state_config.virginia.virginia_consumption_factor import VirginiaPopulationData
from data.population import PopulationData
from data.weather import PyWeatherData, Weather
import datetime
import pandas as pd
from models.seasonality.seasonality import calculate_climatology, calculate_differences_for_df


def calculate_consumption_factor_via_pop_weighted_weather(population_weighted_weather: Weather,
                                                         start_datetime: datetime.datetime,
                                                         end_datetime: datetime.datetime,
                                                         location: str,
                                                         degree_day_type: str = "HDD",
                                                         differencing: bool = True) -> pd.Series:
    """
    Calculates the consumption factor for Prescient Weather.
    """

    if degree_day_type == "CDD":
        df = population_weighted_weather.get_cdd(location,
                                                 start_datetime,
                                                 end_datetime)
    else:
        df = population_weighted_weather.get_hdd(location,
                                                 start_datetime,
                                                 end_datetime)

    if differencing:
        dd_diff_df = calculate_differences_for_df(df, degree_day_type)
        dd_diff_df["Consumption_Factor_Normalizied"] = dd_diff_df["dd_diff"]
        result = dd_diff_df
    else:
        min_value = df[degree_day_type].min()
        max_value = df[degree_day_type].max()
        df["Consumption_Factor_Normalizied"] = df[degree_day_type].apply(lambda x: (x - min_value) / (max_value - min_value))
        date_range = pd.date_range(start=start_datetime, end=end_datetime, freq="D")
        result = pd.DataFrame(date_range, columns=["Date"])
        result["Consumption_Factor_Normalizied"] = df["Consumption_Factor_Normalizied"]

    return result


def calculate_consumption_factor(population: PopulationData,
                                 weather_service: Weather,
                                 start_datetime: datetime.datetime,
                                 end_datetime: datetime.datetime) -> pd.Series:
    """
    Calculates the consumption factor that will be used as a correlation factor
    for natural gas consumption.

    The major themes of this function are:
        1. Grab the population data
        2. Grab the weather data
        3. Calculate the population-weighted weather

    It was also and idea to test out the convergence or lack there of the consumption factor
    with different sets of weather stations. For instance, one can imagine that with just one weather station
    the calculation of the consumption factor will be worse than a calculation with many weather stations.

    """

    population_locations = population.get_locations()

    weather_for_locations = weather_service.get_hdd(population_locations,
                                                    start_datetime,
                                                    end_datetime)

    date_range = pd.date_range(start=start_datetime, end=end_datetime, freq="D")
    population_weighted_hdd = None
    location_to_dfs = dict()
    for location in weather_for_locations:
        pop = population.get_population_for_state_subregion_during_period(location,
                                                                        start_datetime,
                                                                        end_datetime)
        hdd = weather_for_locations[location]
        location_df = pd.DataFrame(date_range, columns=["Date"])
        merged_df = location_df.merge(pop, on="Date").merge(hdd, on="Date")
        if len(merged_df) != len(merged_df.dropna()):
            continue
        else:
            location_to_dfs[location] = merged_df

    result = pd.DataFrame(date_range, columns=["Date"])
    result["Consumption_Factor"] = 0
    for location in location_to_dfs:
        location_df = location_to_dfs.get(location)
        result["Consumption_Factor"] += location_df["Daily_Population"] * location_df["HDD"]

    min_value = result["Consumption_Factor"].min()
    max_value = result["Consumption_Factor"].max()
    result["Consumption_Factor_Normalizied"] = result["Consumption_Factor"].apply(lambda x: (x - min_value) / (max_value - min_value))
    return result












