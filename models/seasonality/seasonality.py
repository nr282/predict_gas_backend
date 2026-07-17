"""
The major goal of the module is to address seasonality.

Seasonality will be addressed via adding of a seasonal time series.

"""

import numpy as np
import pandas as pd
import datetime

def get_day_of_year(date_str: str):
    if isinstance(date_str, str):
        date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    elif isinstance(date_str, datetime.datetime):
        date_obj = date_str
    else:
        date_obj = date_str
    day_of_year = date_obj.timetuple().tm_yday
    return day_of_year

def get_cosine_1_year_period_value(day_of_year):

    return np.cos(2 * np.pi * day_of_year / 365)

def get_cosine_6_month_period_value(day_of_year):

    return np.cos(2 * np.pi * day_of_year / 182)

def get_time_series_1(dates):

    time_vals = list(map(lambda x: get_cosine_1_year_period_value(get_day_of_year(x)), dates))
    time_val = np.array(time_vals)
    return time_vals


def get_time_series_2(dates):

    time_vals = list(map(lambda x: get_cosine_6_month_period_value(get_day_of_year(x)), dates))
    time_val = np.array(time_vals)
    return time_vals

def calculate_climatology(df: pd.DataFrame, degree_day_type: str) -> dict:
    """
    Calculate the climatology for a given degree day type.

    There is an approximation for day 366 for the leap year.

    """

    df["day_of_year"] = df["Date"].apply(lambda x: get_day_of_year(x))
    day_of_year_to_value = df[[degree_day_type, "day_of_year"]].groupby("day_of_year")[degree_day_type].mean()
    avg_dd = day_of_year_to_value.to_dict()
    avg_dd[366] = avg_dd[365]
    return avg_dd


def calculate_differences_for_df(df: pd.DataFrame, degree_day_type: str):

    avg_dd = calculate_climatology(df, degree_day_type)
    df["day_of_year"] = df["Date"].apply(lambda x: get_day_of_year(x))
    df["avg_dd"] = df["day_of_year"].apply(lambda x: avg_dd[x])
    df["dd_diff"] = df[degree_day_type] - df["avg_dd"]
    return df




def example():
    from datetime import datetime

    # Example date: February 7, 2023
    year, month, day = 2023, 2, 7
    date_obj = datetime(year, month, day)
    day_of_year = date_obj.timetuple().tm_yday

    # Get the day of the year
    print(f"The day of the year for {date_obj.strftime('%Y-%m-%d')} is: {day_of_year}")




if __name__ == "__main__":
    example()