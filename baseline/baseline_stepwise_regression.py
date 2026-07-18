"""
The following aims to be the baseline technique that I think is good enough to commercialize.


EIA Data

                -------------> Correlate --------------> Provide the estimates that are relevant to the user
Weather Data

1. If the user is advanced, then we will provide the auxillary conditions. This will be the second version of the api.
2. If the user is not advanced, then we will provide just the natural gas consumption estimates.




"""
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.special import pbdn_seq

from data.weather import PrescientWeather
import datetime
from enum import Enum
from models.seasonality.seasonality import calculate_differences_for_df
from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats
import calendar
from date_utils.date_utils import get_number_days_in_month
from utils_native import get_base_path
import logging
from collections import namedtuple
from typing import Callable
from dataclasses import dataclass
from fdtt.pandas_add_on import solve_pandas_series
import numpy as np
from sklearn.linear_model import LinearRegression

TempInterval = namedtuple('TempInterval', ['startTempIntervalDaily', 'endTempIntervalDaily', "relative_error", "rvalue"])
TimeInterval = namedtuple('TimeInterval', ['start_month', 'end_month', "relative_error", "rvalue"])


class ComponentType(Enum):
    RESIDENTIAL="RESIDENTIAL"
    COMMERCIAL="COMMERCIAL"
    ELECTRIC="ELECTRIC"



@dataclass
class PredictGasResult:
    """
    PredictGasResult provides the output from a linear regression model
    where relevant elements are broken down.

    """

    eia_monthly_values: pd.Series
    eia_normal_values: pd.Series
    eia_monthly_diff: pd.Series
    weather_normal_values: pd.Series
    weather_values: pd.Series
    start_date: datetime
    end_date: datetime
    eia_start_date: datetime
    eia_end_date: datetime
    normal_start_date: datetime
    normal_end_date: datetime
    current_date: datetime
    params_temp: datetime
    params_monthly: dict
    state: str
    component_type: ComponentType
    sensitivity_function: Callable
    consumption_factor_diff: pd.Series
    minimum_consumption: pd.Series
    daily_eia_values: pd.Series
    alpha_daily: pd.Series
    minimum_consumption_daily: pd.Series



component_to_type = dict()
component_to_type[ComponentType.RESIDENTIAL] = "HDD"
component_to_type[ComponentType.COMMERCIAL] = "HDD"
component_to_type[ComponentType.ELECTRIC] = "CDD"


def calculate_consumption_factor_diff(start_date,
                                      end_date,
                                      normal_weather,
                                      weather_values,
                                      component_type: ComponentType):



    weather_values_with_diff = calculate_differences_for_df(weather_values, component_to_type[component_type])
    weather_values_with_diff["diff"] = weather_values_with_diff.apply(lambda row: row[component_to_type[component_type]]
                                                    - normal_weather[row["day_of_year"]], axis=1)
    weather_values_with_diff[component_to_type[component_type]] = weather_values[component_to_type[component_type]]

    weather_values_with_diff["Day"] = weather_values["Date"].apply(lambda x: x.day)
    weather_values_with_diff["Year"] = weather_values["Date"].apply(lambda x: x.year)
    weather_values_with_diff["Month"] = weather_values["Date"].apply(lambda x: x.month)
    return weather_values[["Date", "Year", "Month", "Day", "diff", component_to_type[component_type]]]

def create_weather_values(start_date,
                          end_date,
                          current_date,
                          state,
                          component_type: ComponentType):

    weather = None
    if component_type in [ComponentType.COMMERCIAL, ComponentType.RESIDENTIAL]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_hdd([state], start_date, end_date, current_date)
    elif component_type in [ComponentType.ELECTRIC]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], start_date, end_date, current_date)
    else:
        pass
    return weather

def create_normal_weather_values(normal_start_date,
                                 normal_end_date,
                                 state,
                                 component_type: ComponentType) -> dict:
    """
    Creates the normal weather values.

    """
    current_date = datetime.datetime.now()
    weather = None
    if component_type in [ComponentType.COMMERCIAL, ComponentType.RESIDENTIAL]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_hdd([state], normal_start_date, normal_end_date, current_date)
    elif component_type in [ComponentType.ELECTRIC]:
        prescient_weather = PrescientWeather([state])
        weather = prescient_weather.get_cdd([state], normal_start_date, normal_end_date, current_date)
    else:
        pass

    weather = calculate_differences_for_df(weather, component_to_type[component_type])
    weather_dict = weather[["day_of_year", "avg_dd"]].set_index("day_of_year").to_dict()["avg_dd"]
    return weather_dict




def calculate_consumption_factor_to_eia_sensitivity_monthly_step_wise_on_consumption_factor(eia_start_date,
                                                                                            eia_end_date,
                                                                                            eia_monthly_values,
                                                                                            consumption_factor_values,
                                                                                            state: str,
                                                                                            component_type: ComponentType,
                                                                                            steps = 5,
                                                                                            datapoints_per_regression = 15
                                                                                            ):
    """
    Calculate the consumption factor to eia sensitivity on a month-by-month basis.

    The regression is performed in a stepwise manner on the basis of number of HDD's in a month.


    """


    consumption_factor_by_month = consumption_factor_values.groupby(["Year", "Month"])[["diff", component_to_type[component_type]]].sum().reset_index()

    comparison = eia_monthly_values.merge(consumption_factor_by_month,
                                           on=["Year", "Month"],
                                           how="inner", suffixes=("_eia", "_consumption_factor"))

    comparison_with_nan_dropped = comparison.dropna()
    if len(comparison_with_nan_dropped) > 0.8 * len(comparison):
        comparison = comparison_with_nan_dropped
    else:
        raise RuntimeError("Insufficient data to calculate sensitivity. Too many nans")

    params = dict()

    # Add comparison regression.
    X = comparison["diff_consumption_factor"].values
    y = comparison["diff_eia"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    res = stats.linregress(X_train, y_train)
    global_slope, global_intercept = res.slope, res.intercept
    y_predict = res.intercept + res.slope * X_test

    error = np.sum(np.abs(y_test - y_predict))
    total_mag = np.sum(np.abs(y_test))

    print(f"Full Regression Error divided by total mag is {error / total_mag}")
    params["slope"] = res.slope
    params["intercept"] = res.intercept

    steps = min(steps, len(comparison) // datapoints_per_regression)

    spacing = np.linspace(0, 1, steps)
    prev_ratio = None
    consumption_factor_name = component_to_type[component_type]
    for i, top_ratio in enumerate(spacing):
        if i == 0:
            prev_ratio = top_ratio
            continue

        try:
            start_temp = np.quantile(comparison[consumption_factor_name].values, prev_ratio)
            end_temp = np.quantile(comparison[consumption_factor_name].values, top_ratio)
            prev_ratio = top_ratio

            comparison_temp = comparison[(comparison[consumption_factor_name] >= start_temp) &
                                         (comparison[consumption_factor_name] <= end_temp)]

            X = comparison_temp["diff_consumption_factor"].values
            y = comparison_temp["diff_eia"].values
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            res = stats.linregress(X_train, y_train)
            rvalue = res.rvalue
            y_predict = res.intercept + res.slope * X_test
            error = np.sum(np.abs(y_test - y_predict))
            total_mag = np.sum(np.abs(y_test))
            print(f"Regression with start temp {start_temp / 30} and end temp {end_temp / 30}. Error divided by total mag is {error/total_mag}")
            temp_params = {"slope": res.slope, "intercept": res.intercept}
            temp_interval = TempInterval(start_temp / 30, end_temp / 30, 100 * error/total_mag, rvalue)
            params[temp_interval] = temp_params
        except:
            temp_params = {"slope": global_slope, "intercept": global_intercept}
            temp_interval = TempInterval(start_temp / 30, end_temp / 30, 100 * error / total_mag, rvalue)
            params[temp_interval] = temp_params

    return params

def calculate_consumption_factor_to_eia_sensitivity_monthly_step_wise_on_month(eia_start_date,
                                                                                eia_end_date,
                                                                                eia_monthly_values,
                                                                                consumption_factor_values,
                                                                                state: str,
                                                                                component_type: ComponentType,
                                                                                steps = 5,
                                                                                datapoints_per_regression = 15):
    """
    Calculates consumption-factor to eia sensitivity on a month-by-month basis stepwise on the basis on the month.
    """

    logging.info("Calculating Sensitivity For Step Wise Regression Based On Month")

    consumption_factor_by_month = consumption_factor_values.groupby(["Year", "Month"])[["diff", component_to_type[component_type]]].sum().reset_index()

    comparison = eia_monthly_values.merge(consumption_factor_by_month,
                                           on=["Year", "Month"],
                                           how="inner", suffixes=("_eia", "_consumption_factor"))

    comparison_with_nan_dropped = comparison.dropna()

    #Goal of this function is to drop nan's required for the regression, but to ensure that we still have
    #sufficient data from the regression.
    if len(comparison_with_nan_dropped) > 0.8 * len(comparison):
        comparison = comparison_with_nan_dropped
    else:
        raise RuntimeError("Insufficient data to calculate sensitivity. Too many nans")



    params = dict()

    # Add comparison regression.
    X = comparison["diff_consumption_factor"].values
    y = comparison["diff_eia"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #res = stats.linregress(X_train, y_train, raise=True)
    res = LinearRegression(fit_intercept=True).fit(X_train[:, np.newaxis], y_train[:, np.newaxis])

    global_slope, global_intercept = res.coef_[0,0], res.intercept_[0]
    global_rvalue = res.score(X_train[:, np.newaxis], y_train[:, np.newaxis])
    y_predict = global_intercept + global_slope * X_test

    error = np.sum(np.abs(y_test - y_predict))
    total_mag = np.sum(np.abs(y_test))

    print(f"Full Regression Error divided by total mag is {error / total_mag}")

    params["slope"] = global_slope
    params["intercept"] = global_intercept
    params["rvalue"] = global_rvalue

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    consumption_factor_name = component_to_type[component_type]
    for i, month in enumerate(months):

        if len(X) == 0 or len(y) == 0:
            raise RuntimeError("Data cannot be empty")
        comparison_month = comparison[comparison["Month"] == month]

        try:

            X = comparison_month["diff_consumption_factor"].values
            y = comparison_month["diff_eia"].values

            if len(X) == 0 or len(y) == 0:
                raise RuntimeError("Data cannot be empty")


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            logging.info(f"Number of datapoints for training {month} is: {len(X_train)}")
            logging.info(f"Number of datapoints for testing {month} is: {len(X_test)}")

            # res = stats.linregress(X_train, y_train, raise=True)
            res = LinearRegression(fit_intercept=True).fit(X_train[:, np.newaxis], y_train[:, np.newaxis])
            rvalue = res.score(X_train[:, np.newaxis], y_train[:, np.newaxis])
            month_slope, month_intercept = res.coef_[0,0], res.intercept_[0]
            y_predict = month_intercept + month_slope * X_test

            error = np.sum(np.abs(y_test - y_predict))
            total_mag = np.sum(np.abs(y_test))

            print(f"Regression with month provided by {month}. Error divided by total mag is {error / total_mag}")
            time_params = {"slope": month_slope, "intercept": month_intercept}
            time_interval = TimeInterval(month, month, 100 * error / total_mag, rvalue)

            params[time_interval] = time_params

        except:
            time_params = {"slope": global_slope, "intercept": global_intercept}
            time_interval = TimeInterval(month, month, 100 * error / total_mag, global_rvalue)
            params[time_interval] = time_params

    return params


def calculate_eia_normal_monthly_values(normal_start_date,
                                        normal_end_date,
                                        state,
                                        component_type: ComponentType
                                        ) -> dict:
    """
    Calculate monthly eia values based on consumption factor values.

    """

    component_name = {ComponentType.RESIDENTIAL: "Residential",
                      ComponentType.COMMERCIAL: "Commercial",
                      ComponentType.ELECTRIC: "Electric"}[component_type]

    eia_values = get_eia_consumption_data_in_pivot_format(start_date=normal_start_date,
                                                         end_date=normal_end_date,
                                                         canonical_component_name=component_name,
                                                         create_new_data=False)


    eia_values = eia_values[[state, "period"]]
    eia_values["Month"] = eia_values["period"].apply(lambda x: int(x[-2:]))
    eia_values["Year"] = eia_values["period"].apply(lambda x: int(x[:4]))
    eia_values = eia_values[["Month", state]]
    eia_values[state] = eia_values[state].astype(float)
    eia_month_values = eia_values.groupby(["Month"])[state].mean().to_dict()
    return eia_month_values

def calculate_eia_monthly_values(eia_start_date,
                                 eia_end_date,
                                 state,
                                 component_type: ComponentType) -> pd.DataFrame:

    component_name = {ComponentType.RESIDENTIAL: "Residential",
                      ComponentType.COMMERCIAL: "Commercial",
                      ComponentType.ELECTRIC: "Electric"}[component_type]

    eia_values = get_eia_consumption_data_in_pivot_format(start_date=eia_start_date,
                                                          end_date=eia_end_date,
                                                          canonical_component_name=component_name,
                                                          create_new_data=False)
    eia_values = eia_values[[state, "period"]]
    eia_values["Month"] = eia_values["period"].apply(lambda x: int(x[-2:]))
    eia_values["Year"] = eia_values["period"].apply(lambda x: int(x[:4]))
    eia_values["Day"] = 1
    eia_values["Date"] = eia_values["period"].apply(lambda x: datetime.datetime(year=int(x[:4]),
                                                                                month=int(x[-2:]),
                                                                                day=1))

    eia_values = eia_values[["Date", "Year", "Month", "Day", state]]
    eia_values[state] = eia_values[state].astype(float)
    return eia_values

def calculate_eia_values_diff(start_date,
                             end_date,
                             eia_normal,
                             eia_monthly_values,
                             state):

    eia_monthly_values["diff"] = eia_monthly_values.apply(lambda row: row[state] - eia_normal[row["Month"]] , axis=1)
    return eia_monthly_values


def convert_date_str_to_datetime(date_str):

    if type(date_str) == str:
        return datetime.datetime.strptime(date_str, "%Y-%m-%d")
    else:
        return date_str


def check_if_last_day_in_month(date: datetime.datetime):
    return date.day == calendar.monthrange(date.year, date.month)[1]

def check_preconditions(start_date: str,
                        end_date: str,
                        eia_start_date: str,
                        eia_end_date: str,
                        normal_start_date: str,
                        normal_end_date: str,
                        current_date: str):


    start_datetime = convert_date_str_to_datetime(start_date)
    end_datetime = convert_date_str_to_datetime(end_date)
    eia_start_datetime = convert_date_str_to_datetime(eia_start_date)
    eia_end_datetime = convert_date_str_to_datetime(eia_end_date)
    normal_start_datetime = convert_date_str_to_datetime(normal_start_date)
    normal_end_datetime = convert_date_str_to_datetime(normal_end_date)
    current_datetime = convert_date_str_to_datetime(current_date)

    primary_ordering = (start_datetime <= end_datetime)
    eia_ordering = (eia_start_datetime <= eia_end_datetime)
    normal_ordering = (normal_start_datetime <= normal_end_datetime)
    normal_end_less_than_start = (normal_end_datetime <= start_datetime)
    start_less_than_current = (start_datetime <= current_datetime)
    current_less_than_end = (current_datetime <= end_datetime)
    start_less_than_eia_start = (start_datetime <= eia_start_datetime)
    eia_start_less_than_current = (eia_start_datetime <= current_datetime)

    #Check last day of month.
    end_datetime_last_day_in_month = check_if_last_day_in_month(end_datetime)
    eia_end_datetime_last_day_in_month = check_if_last_day_in_month(eia_end_datetime)
    normal_end_datetime_last_day_in_month = check_if_last_day_in_month(normal_end_datetime)


    precondition = [
        primary_ordering,
        eia_ordering,
        normal_ordering,
        normal_end_less_than_start,
        normal_end_less_than_start,
        start_less_than_current,
        current_less_than_end,
        start_less_than_eia_start,
        start_less_than_current,
        end_datetime_last_day_in_month,
        eia_end_datetime_last_day_in_month,
        normal_end_datetime_last_day_in_month
    ]

    checks = {"start_datetime <= end_datetime": primary_ordering,
              "eia_start_datetime <= eia_end_datetime": eia_ordering,
              "normal_start_datetime <= normal_end_datetime": normal_ordering,
              "normal_end_datetime < start_datetime": normal_end_less_than_start,
              "start_datetime < current_datetime": start_less_than_current,
              "current_datetime < end_datetime": current_less_than_end,
              "start_datetime < eia_start_datetime": start_less_than_eia_start,
              "eia_start_datetime < current_datetime": eia_start_less_than_current,
              "end_datetime_last_day_in_month": end_datetime_last_day_in_month,
              "eia_end_datetime_last_day_in_month": eia_end_datetime_last_day_in_month,
              "normal_end_datetime_last_day_in_month": normal_end_datetime_last_day_in_month}

    return all(precondition), checks


def is_between(date, start_date, end_date):
    return (start_date <= date) and (date <= end_date)


def get_dates_in_month(month, year):

    return pd.date_range(start=datetime.date(year, month, 1),
                         end=datetime.date(year, month, calendar.monthrange(year, month)[1]))

def calculate_eia_daily_value(eia_monthly_value: float,
                              weather_values: pd.Series,
                              date: datetime.datetime,
                              component_type: ComponentType,
                              month,
                              year,
                              params):
    """
    Calculates the daily values given an (1) eia_monthly_value, (2)
    weather_values, (3) date, (4) component_type, (5) month,
    and (6) year.
    """

    if np.isnan(eia_monthly_value) or eia_monthly_value is None or np.isclose(eia_monthly_value, 0):
        raise ValueError("EIA Monthly Value is NaN or None")

    if np.isclose(params["slope"], 0):
        raise ValueError("Slope is 0")


    dates_in_month = get_dates_in_month(month, year)
    weather_dates_in_month = weather_values["Date"].unique()

    if len(dates_in_month) != len(weather_dates_in_month):
        return float('nan')

    year = date.year
    month = date.month
    day = date.day

    slope = params["slope"]
    degree_day_type = component_to_type[component_type]
    weather_values["eia_implied_weather"] = weather_values[degree_day_type].apply(lambda dd: slope * dd)
    missing_eia = eia_monthly_value - weather_values["eia_implied_weather"].sum()
    min_consumption = missing_eia / len(dates_in_month)
    weather_values["eia_daily"] = weather_values[degree_day_type].apply(lambda dd: slope * dd + min_consumption)

    eia_daily_value = weather_values.query(f"Month == {month} "
                                           f"and Year == {year} "
                                           f"and Day == {day}")["eia_daily"].iloc[0]

    if np.isnan(eia_daily_value):
        raise ValueError(f"EIA Daily Value is NaN or None or <= 0 for date: {date}")

    return eia_daily_value


def calculate_eia_daily_values_with_params(eia_monthly_values,
                                            eia_normal_values,
                                            weather_normal_values,
                                            weather_values,
                                            start_date,
                                            end_date,
                                            eia_start_date,
                                            eia_end_date,
                                            normal_start_date,
                                            normal_end_date,
                                            current_date,
                                            params_temp,
                                            params_monthly,
                                            state,
                                            component_type: ComponentType,
                                            sensitivity_function: Callable):
    """
    Calculate eia daily values with params.
    """

    start_date = convert_date_str_to_datetime(start_date)
    end_date = convert_date_str_to_datetime(end_date)
    eia_start_date = convert_date_str_to_datetime(eia_start_date)
    eia_end_date = convert_date_str_to_datetime(eia_end_date)
    current_date = convert_date_str_to_datetime(current_date)

    dates_to_predict = pd.date_range(start=start_date,
                                     end=end_date,
                                     freq="D")

    date_to_value = dict()
    for date in dates_to_predict:
        month = date.month
        year = date.year
        #NOTE: Weather values may not include all weather values for the provided month.
        weather_values_for_month = weather_values.query(f"Month == {month} and Year == {year}")
        temp = weather_values_for_month["HDD"].mean()
        #TODO: This line below should be re-examined.
        #TODO: This function should take in the date also.
        #TODO: The function signature should be: sensitivity_function(date, temp)
        slope, intercept, relative_error = sensitivity_function(month, temp, date)
        import logging
        logging.info(f"Relative Error For this calculation is: {relative_error}")
        params = dict()
        params["slope"] = slope
        params["intercept"] = intercept
        if is_between(date, eia_start_date, eia_end_date):
            eia_monthly_values_for_month = eia_monthly_values.query(f"Month == {month} and Year == {year}")
            if len(eia_monthly_values_for_month) == 1:
                eia_monthly_value = float(eia_monthly_values[state].iloc[0])
                if np.isnan(eia_monthly_value):

                    predicted_eia_monthly_value = (params["slope"] * weather_values_for_month["diff"].sum()
                                                   + params["intercept"] +
                                                   eia_normal_values[month])
                    daily_value = calculate_eia_daily_value(predicted_eia_monthly_value,
                                                            weather_values_for_month,
                                                            date,
                                                            component_type,
                                                            month,
                                                            year,
                                                            params)
                else:
                    if eia_monthly_value is float('nan'):
                        raise ValueError(f"EIA Monthly Value is NaN for month: {month} and year: {year}")

                    daily_value = calculate_eia_daily_value(eia_monthly_value,
                                                            weather_values_for_month,
                                                            date,
                                                            component_type,
                                                            month,
                                                            year,
                                                            params)
            elif len(eia_monthly_values) == 0:
                raise ValueError(f"EIA Monthly Values: There are no values for month: {month} and year: {year}")
            else:
                print(f"EIA Monthly Values: There are more than one: {eia_monthly_values}")
                eia_monthly_value = float(eia_monthly_values[state].iloc[0])

                if np.isnan(eia_monthly_value):
                    predicted_eia_monthly_value = (params["slope"] * weather_values_for_month["diff"].sum()
                                                   + params["intercept"] +
                                                   eia_normal_values[month])
                    daily_value = calculate_eia_daily_value(predicted_eia_monthly_value,
                                                            weather_values_for_month,
                                                            date,
                                                            component_type,
                                                            month,
                                                            year,
                                                            params)
                else:
                    if eia_monthly_value is float('nan'):
                        raise ValueError(f"EIA Monthly Value is NaN for month: {month} and year: {year}")
                    daily_value = calculate_eia_daily_value(eia_monthly_value,
                                                            weather_values_for_month,
                                                            date,
                                                            component_type,
                                                            month,
                                                            year,
                                                            params)

        else:
            predicted_eia_monthly_value = (params["slope"] * weather_values_for_month["diff"].sum()
                                           + params["intercept"] +
                                           eia_normal_values[month])
            daily_value = calculate_eia_daily_value(predicted_eia_monthly_value,
                                                    weather_values_for_month,
                                                    date,
                                                    component_type,
                                                    month,
                                                    year,
                                                    params)


        if np.isnan(daily_value) or np.isclose(daily_value, 0):
            import logging
            logging.info(f"Date is provided by: {date}. The value is provided by {daily_value}")
            raise ValueError(f"Daily Value is NaN or 0 for date: {date}")

        if type(daily_value) != float and type(daily_value) != np.float64:
            if type(daily_value) == np.ndarray:
                daily_value = daily_value[0]

        date_to_value[date] = daily_value



    result = pd.DataFrame.from_dict({"Date": [key for key in date_to_value],
                                     "Value": [date_to_value[key] for key in date_to_value]})

    result["Date"] = result["Date"].astype(str)
    result["Value"] = result["Value"].astype(float)

    return result


def get_params(monthly_hdd, params):
    """
    Calculate the params that are appropiate for a particular HDD level.


    :param hdd:
    :param params:
    :return:
    """

    daily_hdd = monthly_hdd / 30

    slope = None
    for temp_interval in params:

        if type(temp_interval) == str:
            continue

        temp_interval_daily_start = temp_interval.startTempIntervalDaily
        temp_interval_daily_end = temp_interval.endTempIntervalDaily
        relative_error = temp_interval.relative_error

        if daily_hdd >= temp_interval_daily_start and daily_hdd <= temp_interval_daily_end:
            slope = params[temp_interval]["slope"]
            break
        else:
            slope = params[temp_interval]["slope"]

    return slope

def get_params_monthly(month, monthly_params):


    for interval in monthly_params:
        if isinstance(interval, TimeInterval):
            params = monthly_params[interval]
            slope = params["slope"]
            intercept = params["intercept"]
            start_month = interval.start_month
            end_month = interval.end_month
            pct_error = interval.relative_error
            if start_month == end_month:
                pass
            else:
                break

            if np.isclose(start_month, month):
                return slope

    return None

def calculate_non_weather_dependent_component(params_time,
                                              params_temp,
                                              weather_values,
                                              eia_values,
                                              component_type: ComponentType,
                                              state: str,
                                              use_stepwise=True):
    """
    Calculates the time varying minimum consumption for a given component type.

    The natural gas consumption is the addition of:
        (1) time-dependent minimum consumption
        (2) weather-dependent consumption

    It will look to calculate the monthly time series for monthly consumption and will
    look to state what percentage non-weather dependent component is of the total consumption.

    Non-weather dependent consumption is around the 50% level.



    """

    try:
        weather_values_by_month = weather_values.groupby(["Year", "Month"])["HDD"].sum().reset_index()
        weather_values_by_month["Consumption"] = weather_values_by_month["HDD"].apply(lambda hdd: hdd * get_params(hdd, params_temp) if use_stepwise else hdd * params_temp["slope"])
        weather_values_by_month.merge(eia_values, on=["Year", "Month"], how="left")
        gas_consumption_comparison = weather_values_by_month.merge(eia_values, on=["Year", "Month"], how="left")
        gas_consumption_comparison["non_weather_dependent_consumption"] = gas_consumption_comparison[state] - gas_consumption_comparison["Consumption"]
        gas_consumption_comparison.plot(x="Date", y="non_weather_dependent_consumption", figsize=(12,12))
        plt.xlabel("Date")
        plt.ylabel("Consumption (MMCF)")
        plt.title(f"Non-Weather Dependent Consumption Over Time For State {state}")
        plt.savefig(f"non_weather_dependent_consumption_{state}.png")
        plt.clf()

        average_non_weather_dependent_consumption = gas_consumption_comparison.groupby("Month")["non_weather_dependent_consumption"].mean().reset_index()
        gas_consumption_comparison = gas_consumption_comparison.merge(average_non_weather_dependent_consumption, on="Month", how="left", suffixes=("", "_average"))
        gas_consumption_comparison["error"] = gas_consumption_comparison["non_weather_dependent_consumption"] - gas_consumption_comparison["non_weather_dependent_consumption_average"]
        gas_consumption_comparison_average_error = gas_consumption_comparison.groupby("Month")["error"].mean().reset_index()
        gas_consumption_comparison.plot(x="Date", y="error", figsize=(12, 12))
        plt.title(f"Error Between Non Weather Dependent Consumption and Average Non Weather Dependent Consumption Over Time For State {state}")
        plt.ylabel("Consumption (MMCF)")
        plt.savefig(f"error_term_{state}.png")
        plt.clf()

        gas_consumption_comparison["pct_weather_dependent_consumption"] = gas_consumption_comparison["non_weather_dependent_consumption"].abs() / gas_consumption_comparison[state]
        gas_consumption_comparison.plot(x="Date", y="pct_weather_dependent_consumption", figsize=(12, 12))
        plt.title(f"Percentage of Non-Weather Dependent Consumption Over Time For State {state}")
        plt.ylabel("Percentage")
        plt.savefig(f"pct_non_weather_dependent_consumption_{state}.png")
        plt.clf()


        average_pct_natural_gas_consumption = gas_consumption_comparison["pct_weather_dependent_consumption"].mean()

        logging.info("Average Percent Non-Weather Dependent Consumption: " + str(average_pct_natural_gas_consumption))
        #For Virginia, non-weather dependent consumption amounts to around 47 percent error of the total natural
        #gas consumption.


        ####################################################################################################################
        ####################################################################################################################
        # Analysis by parameter monthly.
        weather_values_by_month = weather_values.groupby(["Year", "Month"])["HDD"].sum().reset_index()
        weather_values_by_month["Consumption"] = weather_values_by_month.apply(
            lambda row: row["HDD"] * get_params_monthly(row["Month"], params_time) if use_stepwise else row["HDD"] * params_time["slope"], axis=1)

        gas_consumption_comparison = weather_values_by_month.merge(eia_values, on=["Year", "Month"], how="left")
        gas_consumption_comparison["non_weather_dependent_consumption"] = gas_consumption_comparison[state] - \
                                                                          gas_consumption_comparison["Consumption"]
        gas_consumption_comparison.plot(x="Date", y="non_weather_dependent_consumption", figsize=(12, 12))
        plt.xlabel("Date")
        plt.ylabel("Consumption (MMCF)")
        plt.title(f"Non-Weather Dependent Consumption Over Time For State {state}")
        plt.savefig(f"non_weather_dependent_consumption_{state}_time_regression.png")
        plt.clf()

        average_non_weather_dependent_consumption = gas_consumption_comparison.groupby("Month")[
            "non_weather_dependent_consumption"].mean().reset_index()
        gas_consumption_comparison = gas_consumption_comparison.merge(average_non_weather_dependent_consumption, on="Month",
                                                                      how="left", suffixes=("", "_average"))
        gas_consumption_comparison["error"] = gas_consumption_comparison["non_weather_dependent_consumption"] - \
                                              gas_consumption_comparison["non_weather_dependent_consumption_average"]
        gas_consumption_comparison_average_error = gas_consumption_comparison.groupby("Month")["error"].mean().reset_index()
        gas_consumption_comparison.plot(x="Date", y="error", figsize=(12, 12))
        plt.title(
            f"Error Between Non Weather Dependent Consumption and Average Non Weather Dependent Consumption Over Time For State {state}")
        plt.ylabel("Consumption (MMCF)")
        plt.savefig(f"error_term_{state}_time_regression.png")
        plt.clf()

        gas_consumption_comparison["pct_weather_dependent_consumption"] = gas_consumption_comparison[
                                                                              "non_weather_dependent_consumption"].abs() / \
                                                                          gas_consumption_comparison[state]
        gas_consumption_comparison.plot(x="Date", y="pct_weather_dependent_consumption", figsize=(12, 12))
        plt.title(f"Percentage of Non-Weather Dependent Consumption Over Time For State {state}")
        plt.ylabel("Percentage")
        plt.savefig(f"pct_non_weather_dependent_consumption_time_regression_{state}.png")
        plt.clf()

        average_pct_natural_gas_consumption = gas_consumption_comparison["pct_weather_dependent_consumption"].mean()

        logging.info("Average Percent Non-Weather Dependent Consumption: " + str(average_pct_natural_gas_consumption))

        #TODO: Apply Variational Mode Decomposition (VMD) to improve forecasting.
        #TODO: https://ww3.math.ucla.edu/camreport/cam13-22.pdf
        #TODO: https://github.com/vrcarva/vmdpy

    except:
        weather_values_by_month = None


    return weather_values_by_month




def apply_fast_disaggregation_in_time_technology(monthly_ts: pd.DataFrame,
                                                 date: datetime.datetime):
    """
    Applies the fast disaggregation in time technology (FDTT) to
    the calculation of the sensitivity for weather.
    """

    year = date.year
    monthly_ts["Date"] = monthly_ts["date"].apply(lambda x: datetime.datetime(year, x, 1))
    AVERAGE_DAYS_IN_MONTH = 30
    monthly_ts["Value"] = AVERAGE_DAYS_IN_MONTH * monthly_ts["month"]
    monthly_ts.drop(columns=["date", "month"], inplace=True)


    fw, corrected_df = solve_pandas_series(monthly_ts, date_is_beginning=True)
    day_of_year = date.timetuple().tm_yday
    value = fw.get_function_evaluated_at_time(day_of_year, 1)
    return value

def create_weather_sensitivity_function(params_temp,
                                        params_time):
    """
    Calculates the sensitivity of the natural gas consumption to temperature, with the relevant
    regressions provided above.

    Articulated drawbacks of this sensitivity function:
        1. The sensitivity function is discontinuous.
        2. The sensitivity function is decoupled.
        It does not combine the (a) temperature and (b) time calculation
        into one sensitivity prediction. The calculation is decoupled.

    It will be critical to visualize this function.

    TODO: Use FDTT (Fast Disaggregation in Time Technology) to solve problem.

    """

    def sensitivity_function(month, temp, date):

        months = []
        alpha_vals = []
        intercept_vals = []

        #Develop the alpha_vals and intercept values
        for time_interval in params_time:
            if not isinstance(time_interval, TimeInterval):
                continue
            start_month = time_interval.start_month
            end_month = time_interval.end_month
            slope = params_time[time_interval]["slope"]
            intercept = params_time[time_interval]["intercept"]
            months.append(start_month)
            alpha_vals.append(slope)
            intercept_vals.append(intercept)

        #Form into time series.
        alpha_ts = pd.DataFrame({"date": months, "month": alpha_vals})
        intercept_vals = pd.DataFrame({"date": months, "month": intercept_vals})
        for time_interval in params_time:
            if not isinstance(time_interval, TimeInterval):
                continue
            start_month = time_interval.start_month
            end_month = time_interval.end_month
            slope = params_time[time_interval]["slope"]
            intercept = params_time[time_interval]["intercept"]
            if np.isclose(start_month, month):
                if time_interval.relative_error < 70:
                    slope_value = apply_fast_disaggregation_in_time_technology(alpha_ts, date)
                    intercept_value = apply_fast_disaggregation_in_time_technology(intercept_vals, date)
                    return slope_value, intercept_value, time_interval.relative_error
                else:
                    break


        result_slope = None
        result_intercept = None
        result_relative_error = None
        for temp_interval in params_temp:
            if not isinstance(temp_interval, TempInterval):
                continue
            slope = params_temp[temp_interval]["slope"]
            intercept = params_temp[temp_interval]["intercept"]
            temp_start = temp_interval.startTempIntervalDaily
            temp_end = temp_interval.endTempIntervalDaily

            if temp_start <= temp <= temp_end:
                result_slope, result_intercept = slope, intercept
                result_relative_error = temp_interval.relative_error
                break
            else:
                result_slope, result_intercept = slope, intercept
                result_relative_error = temp_interval.relative_error

        return result_slope, result_intercept, result_relative_error

    return sensitivity_function


def build_predict_gas(eia_monthly_values,
                        eia_normal_values,
                        eia_monthly_diff,
                        weather_normal_values,
                        weather_values,
                        start_date,
                        end_date,
                        eia_start_date,
                        eia_end_date,
                        normal_start_date,
                        normal_end_date,
                        current_date,
                        params_temp,
                        params_monthly,
                        state,
                        component_type,
                        sensitivity_function,
                        consumption_factor_diff,
                        minimum_consumption,
                        daily_eia_values) -> PredictGasResult:
    """
    Builds the predict gas result.

    """

    predict_gas_result = PredictGasResult(eia_monthly_values,
                                          eia_normal_values,
                                          eia_monthly_diff,
                                          weather_normal_values,
                                          weather_values,
                                          start_date,
                                          end_date,
                                          eia_start_date,
                                          eia_end_date,
                                          normal_start_date,
                                          normal_end_date,
                                          current_date,
                                          params_temp,
                                          params_monthly,
                                          state,
                                          component_type,
                                          sensitivity_function,
                                          consumption_factor_diff,
                                          minimum_consumption,
                                          daily_eia_values,
                                          pd.Series([]),
                                          pd.Series([]))


    return predict_gas_result





def calculate_eia_daily_values(start_date: str,
                               end_date: str,
                               eia_start_date: str,
                               eia_end_date: str,
                               normal_start_date: str,
                               normal_end_date: str,
                               current_date: str,
                               component_type: ComponentType,
                               state) -> PredictGasResult:
    """
    Calculates the eia daily values.

    The following diagram lays out how I would like the dates to divide time.

    There are 7 dates that are laid out above:
        1. start_date (std)
        2. end_date (etd)
        3. eia_start_date (eia_std)
        4. eia_end_date (eia_etd)
        5. normal_start_date (n_std)
        6. normal_end_date (n_etd)
        7. current_date (c_d)

                                        Timeline
        --------------------------------------------------------------------------------
            |           |       |     |            |      |                            |
        (n_std)     (n_etd)   (std)   |            |    (c_d)                        (etd)
                                      |            |
                                    (eia_std)     (eia_etd)


    Constraints:

        Normalization Dates must begin before the start date (std) for which we want to find the daily
        values. Hence, n_etd >= n_std, and n_etd < std.

        Likewise, EIA values must occur before the current date marker. Hence, eia_etd >= std, and eia_etd < c_d.

        Likewise, c_d <= etd.

    """

    #################################################################
    ################# CHECK PRECONDITIONS ###########################

    preconditions_satisfied, checks = check_preconditions(start_date,
                                                        end_date,
                                                        eia_start_date,
                                                        eia_end_date,
                                                        normal_start_date,
                                                        normal_end_date,
                                                        current_date)

    if not preconditions_satisfied:
        raise ValueError(f"Preconditions not satisfied. Checks are provided by: "
                         f"{checks}")

    #################################################################
    #Begin Weather Calculation
    #Create normal weather.
    weather_normal_values = create_normal_weather_values(normal_start_date,
                                                         normal_end_date,
                                                         state,
                                                         component_type)

    assert(type(weather_normal_values) == dict)

    #Get weather values.
    weather_values = create_weather_values(start_date,
                                           end_date,
                                           current_date,
                                           state,
                                           component_type)

    assert(len(weather_values) == len(pd.date_range(start=start_date, end=end_date)))
    assert(not weather_values[component_to_type[component_type]].isna().any())

    #Step 1. Calculate consumption factor normal values.
    consumption_factor_diff = calculate_consumption_factor_diff(start_date,
                                                                end_date,
                                                                weather_normal_values,
                                                                weather_values,
                                                                component_type)

    assert("Date" in consumption_factor_diff.columns)

    #################################################################
    #Begin EIA Calculation
    # Step 3: Calculate eia normal values.
    eia_normal_values = calculate_eia_normal_monthly_values(normal_start_date,
                                                          normal_end_date,
                                                          state,
                                                          component_type)

    assert(type(eia_normal_values) == dict)


    #Step 3: Calculate eia monthly values.
    eia_monthly_values = calculate_eia_monthly_values(eia_start_date,
                                                      eia_end_date,
                                                      state,
                                                      component_type)


    eia_monthly_diff = calculate_eia_values_diff(eia_start_date,
                                                eia_end_date,
                                                eia_normal_values,
                                                eia_monthly_values,
                                                 state)

    assert("Date" in eia_monthly_diff.columns)


    #################################################################
    #Calculate sensitivity.
    #Step 3. Calculate sensitivity between eia monthly,
    #and consumption factor values, conditioned on temperature
    params_temp = calculate_consumption_factor_to_eia_sensitivity_monthly_step_wise_on_consumption_factor(eia_start_date,
                                                                                                     eia_end_date,
                                                                                                     eia_monthly_diff,
                                                                                                     consumption_factor_diff,
                                                                                                     state,
                                                                                                     component_type)


    #################################################################
    # Calculate sensitivity.
    # Step 3. Calculate sensitivity between eia monthly,
    # and consumption factor values conditioned on month.
    params_monthly = calculate_consumption_factor_to_eia_sensitivity_monthly_step_wise_on_month(eia_start_date,
                                                                                                 eia_end_date,
                                                                                                 eia_monthly_diff,
                                                                                                 consumption_factor_diff,
                                                                                                 state,
                                                                                                 component_type)

    ###############################################################
    #Calculate non-weather dependent component, which is called the
    #time-varying minimum consumption.
    minimum_consumption = calculate_non_weather_dependent_component(params_monthly,
                                                                    params_temp,
                                                                    weather_values,
                                                                    eia_monthly_values,
                                                                    component_type,
                                                                    state)

    ###############################################################
    #Use (1) time based parameters and (2) temperature-based parameters to develop a way
    #of calculating a function that provides intercept/slope information on the basis
    #of time and temperature.
    sensitivity_function = create_weather_sensitivity_function(params_temp, params_monthly)


    #################################################################
    #Calculate EIA Daily Values.
    #Step 4: For dates in which no eia monthly exists, apply
    #sensitivity to weather. For dates, in which eia monthly
    #dates exist, form via weather calculation.
    daily_eia_values = calculate_eia_daily_values_with_params(eia_monthly_values,
                                                            eia_normal_values,
                                                            weather_normal_values,
                                                            weather_values,
                                                            start_date,
                                                            end_date,
                                                            eia_start_date,
                                                            eia_end_date,
                                                            normal_start_date,
                                                            normal_end_date,
                                                            current_date,
                                                            params_temp,
                                                            params_monthly,
                                                            state,
                                                            component_type,
                                                            sensitivity_function)

    daily_eia_values["Value"] = daily_eia_values["Value"].ewm(com=3).mean()


    ####################################################################################################################
    ################################################# Build PredictGasResult ###########################################

    predict_gas_result = build_predict_gas(eia_monthly_values,
                                            eia_normal_values,
                                            eia_monthly_diff,
                                            weather_normal_values,
                                            weather_values,
                                            start_date,
                                            end_date,
                                            eia_start_date,
                                            eia_end_date,
                                            normal_start_date,
                                            normal_end_date,
                                            current_date,
                                            params_temp,
                                            params_monthly,
                                            state,
                                            component_type,
                                            sensitivity_function,
                                            consumption_factor_diff,
                                            minimum_consumption,
                                            daily_eia_values)

    return predict_gas_result



if __name__ == "__main__":

    result = calculate_eia_daily_values("2009-01-01",
                               "2025-09-30",
                               "2009-01-01",
                               "2025-09-30",
                               "2000-01-01",
                               "2008-12-31",
                               "2025-09-30",
                               ComponentType.RESIDENTIAL,
                               "New York")
