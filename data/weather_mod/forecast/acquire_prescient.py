"""
Acquire Prescient Weather Data via this module.

The module will aim to acquire the Prescient Weather Data.
"""

import requests
import pandas as pd
from typing import TypeVar, TypedDict
from data.eia_consumption.eia_geography_mappings import us_state_to_abbrev
import datetime


class PrescientWeatherDataResponse(TypedDict):
    region: str
    variable: str
    model: str
    initdate: str
    forecast: list
    forecast: dict



allowable_degree_day_types = ["popcdd",
                              "eleccdd",
                              "gascdd",
                              "oilcdd",
                              "pophdd",
                              "elechdd",
                              "gashdd",
                              "oilhdd",
                              "tdd"]

allwoable_regions = ["CONUS",
                     "ELECTRIC",
                     "EIAGAS",
                     "CENSUS",
                     "CAISO",
                     "ERCOT",
                     "ERCOT_AUSTIN",
                     "ERCOT_CPS",
                     "ERCOT_HOUSTON",
                     "ERCOT_NORTH",
                     "ERCOT_SOUTH",
                     "ERCOT_WEST",
                     "ERCOT_WX_COAST",
                     "ERCOT_WX_EAST",
                     "ERCOT_WX_FARWEST",
                     "ERCOT_WX_NORTH",
                     "ERCOT_WX_NORTHCENTRAL",
                     "ERCOT_WX_SOUTH",
                     "ERCOT_WX_SOUTHCENTRAL",
                     "ERCOT_WX_WEST",
                     "ISO-NE",
                     "MISO",
                     "NYISO",
                     "NW",
                     "PJM",
                     "SE",
                     "SPP",
                     "SW",
                     "EIA EAST",
                     "EIA MIDWEST",
                     "EIA MOUNTAIN",
                     "EIA PACIFIC",
                     "EIA SOUTH CENTRAL",
                     "EAST",
                     "E N CENTRAL",
                     "E S CENTRAL",
                     "MIDDLE ATLANTIC",
                     "MOUNTAIN",
                     "NEW ENGLAND",
                     "PACIFIC",
                     "SOUTH ATLANTIC",
                     "W N CENTRAL",
                     "W S CENTRAL",
                     "AR",
                     "AZ",
                     "CA",
                     "CO",
                     "CT",
                     "DE",
                     "FL",
                     "GA",
                     "IA",
                     "ID",
                     "IL",
                     "IN",
                     "KS",
                     "KY",
                     "LA",
                     "MA",
                     "MD",
                     "ME",
                     "MI",
                     "MN",
                     "MO",
                     "MS",
                     "MT",
                     "NC",
                     "ND",
                     "NE",
                     "NH",
                     "NJ",
                     "NM",
                     "NV",
                     "NY",
                     "OH",
                     "OK",
                     "OR",
                     "PA",
                     "RI",
                     "SC",
                     "SD",
                     "TN",
                     "TX",
                     "UT",
                     "VA",
                     "VT",
                     "WA",
                     "WI",
                     "WV",
                     "WY"]


def acquire_prescient_weather(region: str,
                              degree_day_type: str,
                              init_date: str):

    print(f"Acquiring weather data for region: {region}")
    bearer_token = "JV55O8qHDHKRr1pReHCexxIAc0DL7DAG"
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "accept": "application/json"  # Optional: Include if sending JSON data
    }

    response = requests.get(f"https://fastapi.worldclimateservice.com/tm-api/v3/forecast/degreeday/mediumrange/daily/ecmwf00z/{degree_day_type}/{region}?climo=10&numfcst=1&initdate={init_date}",
                            headers=headers)


    return response

def parse_prescient_weather_data_response(response) -> PrescientWeatherDataResponse:

    if type(response) == list:
        if type(response[0]) == dict:
            response = response[0]
    elif type(response) == dict:
        pass

    response_dict: PrescientWeatherDataResponse = {'region': response.get('region'),
                                                   'variable': response.get('variable'),
                                                   'model': response.get('model'),
                                                   'initdate': response.get('initdate'),
                                                   'forecast': response.get('forecast')}

    return response_dict


def parse_response(prescient_weather_response: PrescientWeatherDataResponse,
                   variable: str) -> pd.DataFrame:
    """
    Parses raw response from Prescient API into a dataframe.

    :param response:
    :return:
    """

    forecast = prescient_weather_response.get("forecast")
    base_date = prescient_weather_response.get("initdate")
    variable = prescient_weather_response.get("variable")
    region = prescient_weather_response.get("region")
    n = len(forecast)
    records = dict()
    records["fcstdate"] = []
    records[variable] = []
    records["initdate"] = []
    records["region"] = []
    for i in range(n):
        forecast_date = forecast[i]["fcstdate"]
        forecast_value = forecast[i][f"{variable}"]
        records["fcstdate"].append(forecast_date)
        records[variable].append(forecast_value)
        records["initdate"].append(base_date)
        records["region"].append(region)

    df = pd.DataFrame(records)
    df["initdate"] = pd.to_datetime(df["initdate"])
    df["fcstdate"] = pd.to_datetime(df["fcstdate"])

    return df

def get_weather_data_for_state(init_date: str, region: str, degree_day_type: str) -> pd.DataFrame:

    if isinstance(init_date, datetime.datetime):
        init_date = datetime.datetime.strftime(init_date, "%Y-%m-%d")

    response = acquire_prescient_weather(region, degree_day_type, init_date)
    if response.status_code != 200 or response.text == "":
        raise RuntimeError(f"Not able to acquire state {region}. Response status code is: {response.status_code}. Response text is: {response.text}.")
    response = response.json()
    response = parse_prescient_weather_data_response(response)
    df = parse_response(response, degree_day_type)

    if len(df) != len(df.dropna()):
        raise ValueError(f"Nans exist in the dataframe for the"
                         f" dataframe parameterizied around"
                         f" {init_date} and region {region} and degree day type {degree_day_type}")



    return df

def get_weather_data_for_all_states(init_date: str, degree_day_type: str):

    df = pd.DataFrame([])
    for state, abbrv in us_state_to_abbrev.items():
        state_df = get_weather_data_for_state(init_date, abbrv, degree_day_type)
        df = pd.concat([df, state_df])
    return df


if __name__ == "__main__":

    df = get_weather_data_for_all_states("2025-02-01", "popcdd")

