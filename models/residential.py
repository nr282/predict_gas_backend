"""
Provides residential model that can be fitted and inferred.

TODO: Added global optimization.
TODO: Global optimization is critical.

TODO: What do I need to accomplish:
    TODO: 1. Remove all parameters except

"""

import datetime
from abc import ABC

import pandas as pd
import numpy as np
from calibration.calibration import calibration
from data.consumption_factor.consumption_factor_calculation import (calculate_consumption_factor,
                                                                    calculate_consumption_factor_via_pop_weighted_weather)
from optimization import grid_search
import pickle
from data.consumption_factor import consumption_factor_calculation
from data.state_config.virginia.virginia_consumption_factor import VirginiaPopulationData
from data.weather import PyWeatherData, PrescientWeather
from data.eia_consumption.eia_consumption import get_eia_consumption_data_in_pivot_format
from models.seasonality.seasonality import get_time_series_1, get_time_series_2
import logging
import pymc as pm
from models.model import Model
from typing import Tuple
import calendar
import os

def map_date_to_index(consumption_factor: pd.DataFrame):
    """
    Map the date to an index and map the index to a date.

    :return:
    """

    consumption_factor_date_to_index = dict()
    index_to_consumption_factor_date = dict()
    for index, row in consumption_factor.iterrows():
        dt = row["Date"]
        consumption_factor_date_to_index[dt] = index
        index_to_consumption_factor_date[index] = dt
    return consumption_factor_date_to_index, index_to_consumption_factor_date, consumption_factor_date_to_index.keys()


class ResidentialModel(Model):
    """
    States the Residential Model that will be used to calculate natural gas
    consumption.

    """

    def __init__(self,
                 calibrated_parameters=None,
                 parameter_list=None,
                 ):

        super().__init__(calibrated_parameters, parameter_list)


    def _calculate_estimated_eia_monthly_data(self, idata):
        """
        Calculates the estimated eia monthly data via the relevant model
        that was sampled.

        :param idata:
        :return:
        """

        eia_observations = idata.posterior.eia_observations
        eia_observations_df = eia_observations.to_dataframe()
        eia_observations_df = eia_observations_df.reset_index()
        eia_observations_by_date = eia_observations_df.groupby(["dates"])["eia_observations"].mean().to_frame()
        eia_observations_by_date["Day"] = eia_observations_by_date.index.to_series().apply(lambda x: x.day)
        eia_observations_by_date["Month"] = eia_observations_by_date.index.to_series().apply(lambda x: x.month)
        eia_observations_by_date["Year"] = eia_observations_by_date.index.to_series().apply(lambda x: x.year)
        eia_observations_by_date["Date"] = eia_observations_by_date.index.to_series().apply(lambda x: datetime.datetime(x.year, x.month, x.day))
        eia_observations_by_month = eia_observations_by_date.groupby(["Year", "Month"])["eia_observations"].sum().reset_index()
        eia_observations_by_month["Date"] = pd.to_datetime(eia_observations_by_month.apply(lambda row: datetime.datetime(int(row["Year"]), int(row["Month"]), 1), axis=1))
        return eia_observations_by_date, eia_observations_by_month

    def inference(self,
                start_datetime: str,
                end_datetime: str,
                eia_start_datetime: str,
                eia_end_datetime: str,
                params: dict,
                data: dict,
                app_params: dict = None):
        """
        Inference in the residential model.

        """

        dates = pd.date_range(start_datetime, end_datetime)
        consumption_factor_values = data["consumption_factor_values"]["Consumption_Factor_Normalizied"].values
        consumption_factor_lagged_values = calculate_consumption_lagged(consumption_factor_values)
        mean_consumption_factor = np.mean(consumption_factor_values)
        variance_consumption_factor = np.var(consumption_factor_values)
        eia_monthly_values = data["eia_monthly_values"]
        full_eia_data = data["full_eia_data"]

        coords = {
            "dates": list(dates),
        }
        
        with pm.Model(coords=coords) as model:

            consumption_factor = pm.Normal("consumption_factor",
                                           mu=mean_consumption_factor,
                                           sigma=variance_consumption_factor,
                                           observed=consumption_factor_values.astype(np.float32),
                                           dims="dates")

            consumption_factor_lagged = pm.Normal("consumption_factor_lagged",
                                                  mu=mean_consumption_factor,
                                                  sigma=variance_consumption_factor,
                                                  observed=consumption_factor_lagged_values.astype(np.float32),
                                                  dims="dates")

            logging.debug("Parameters are provided by: {params}".format(params=params))

            #Alpha is a measure of sensitivity to weather
            alpha = pm.Normal("alpha_1",
                              mu=float(params.get("alpha_mu")),
                              sigma=float(params.get("alpha_sigma")))

            alpha_2 = pm.Normal("alpha_2", mu=float(params.get("alpha_2_mu")), sigma=float(params.get("alpha_2_sigma")))


            eia_daily_observations = pm.Normal("eia_observations",
                                               mu=(alpha + alpha_2) * consumption_factor + alpha_2 * consumption_factor_lagged,
                                               sigma=params.get("daily_consumption_error"),
                                               dims="dates")

            consumption_factor_to_index, index_to_consumption_factor_date, dates = map_date_to_index(data["consumption_factor_values"])

            calculate_eia_monthly_consumption_constraints(model,
                                                          eia_daily_observations,
                                                          full_eia_data,
                                                          consumption_factor_to_index,
                                                          data["state"],
                                                          eia_monthly_start_date=eia_start_datetime,
                                                          eia_monthly_end_date=eia_end_datetime,
                                                          sigma=params.get("monthly_consumption_error"),
                                                          app_params=app_params)

            try:
                idata = pm.sample(draws=20, tune=20, cores=1)
            except:
                return None, None, None
            eia_estimated_daily_observations, estimated_estimated_monthly_data = self._calculate_estimated_eia_monthly_data(idata)
            return eia_estimated_daily_observations, estimated_estimated_monthly_data, params


    def inference_for_daily_values(self,
                                   state,
                                   start_datetime: str,
                                   end_datetime: str,
                                   data: dict,
                                   daily_adjustments: pd.Series,
                                   app_params: dict = None):
        """
        Inference for daily values for the given state, start time, end time,
        data and the app params.

        The goal is to calculate the daily consumption values for the given
        state during the given period of time, by applying the data and the daily
        adjustments.

        Add (1) daily_adjustments and (2) eia_daily_average, add the two.
        """

        eia_daily_average = data["eia_average_daily_values"]
        eia_all_values = eia_daily_average.merge(daily_adjustments, how="left", on="Date")
        eia_all_values["eia_daily_consumption"] = eia_daily_average["eia_daily_average"] + eia_all_values["eia_observations"]
        return eia_all_values[["Date", "eia_daily_consumption"]]

    def get_params_for_model(self) -> dict:
        """
        Calculates parameters for the model.

        The parameters that will be used in the model are:
            1. alpha_mu
            2. alpha_sigma
            3. alpha_2_mu
            4. alpha_2_sigma
            7. daily_consumption_error

        """

        return {"alpha_mu": 0,
                "alpha_sigma": 10,
                "alpha_2_mu": 0,
                "alpha_2_sigma": 1,
                "daily_consumption_error": 0,
                "monthly_consumption_error": 0.0
                }


#Very similar to how we specified the Residential Model below,
#we need to specify a Residential Model that uses the linear regression.
class ResidentialModelLinearRegression(Model):
    """
    States the linear regression-based Residential Model.

    The linear regression-based model aims to create daily values
    via the data that is provided.

    The goal is to compare the accuracy of these data values versus the
    values that were provided above.
    """

    def __init__(self,
                 calibrated_parameters=None,
                 parameter_list=None,
                 ):

        super().__init__(calibrated_parameters, parameter_list)


    def inference(self,
                  start_datetime: str,
                  end_datetime: str,
                  eia_start_datetime: str,
                  eia_end_datetime: str,
                  params: dict,
                  data: dict):
        """
        Inference in the residential model.

        Returns (1) params, (2) eia_estimated_daily_observations, (3) estimated_monthly_data.

        """

        dates = pd.date_range(start_datetime, end_datetime)
        consumption_factor_values = data["consumption_factor_values"]["Consumption_Factor_Normalizied"].values

        df = pd.DataFrame(data={"Date": dates,
                                "Consumption_Factor_Values": consumption_factor_values})


        df["Month"] = df["Date"].dt.month
        df["Year"] = df["Date"].dt.year

        df_by_month = df.groupby(["Month", "Year"]).mean().reset_index()

        full_eia_data = data["full_eia_data"]

        full_eia_data = full_eia_data.reset_index()
        full_eia_data["Date"] = full_eia_data["period"].apply(lambda x: x + "-01")
        full_eia_data["Date"] = full_eia_data["Date"].apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

        full_eia_data["Day"] = full_eia_data["Date"].apply(lambda x: x.day)
        full_eia_data["Month"] = full_eia_data["Date"].apply(lambda x: x.month)
        full_eia_data["Year"] = full_eia_data["Date"].apply(lambda x: x.year)

        merged_df = df_by_month.merge(full_eia_data, how="left", on=["Month", "Year"]).dropna()
        merged_df["Virginia"] = merged_df["Virginia"].astype(float)
        merged_df_fit = merged_df[merged_df["Date_y"] <= eia_end_datetime][["Consumption_Factor_Values", "Virginia", "Month", "Year"]]

        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(merged_df_fit["Consumption_Factor_Values"].values[..., np.newaxis], merged_df_fit["Virginia"].values[..., np.newaxis])
        alpha = reg.coef_[0][0]
        beta = reg.intercept_[0]
        merged_df["alpha"] = alpha
        merged_df["beta"] = beta
        merged_df["Monthly_Predicted"] = merged_df.apply(lambda row: row["alpha"] * row["Consumption_Factor_Values"] + row["beta"], axis=1)
        merged_df["error"] = (merged_df["Monthly_Predicted"] - merged_df["Virginia"]).abs() / merged_df["Virginia"]
        relative_error = merged_df["error"].mean()
        percent_error = relative_error * 100

        logging.info("Percent Error of Linear Regression Model is {}".format(percent_error))
        #INFO:root:Percent Error of Linear Regression Model is 22.3%

        eia_estimated_daily_observations = None
        estimated_monthly_data = None
        params = None

        return eia_estimated_daily_observations, estimated_monthly_data, params

    def get_params_for_model(self) -> dict:
        """
        Calculates parameters for the model.

        The parameters that will be used in the model are:
            1. alpha_mu
            2. alpha_sigma
            3. alpha_2_mu
            4. alpha_2_sigma
            5. daily_consumption_error

        """

        return {"alpha_mu": 0,
                "alpha_sigma": 10,
                "alpha_2_mu": 0,
                "alpha_2_sigma": 1,
                "daily_consumption_error": 0,
                "monthly_consumption_error": 0.0
                }


def calculate_consumption_lagged(consumption_factor_values):
    """
    Calculate consumption factor lagged values.
    """

    consumption_factor_values_lagged = np.roll(consumption_factor_values, shift=1)
    val = consumption_factor_values_lagged[1]
    consumption_factor_values_lagged[0] = val
    return consumption_factor_values_lagged

def is_data_between_dates(eia_monthly_start_date, eia_monthly_end_date, current_month):

    return (eia_monthly_start_date <= current_month) and (current_month <= eia_monthly_end_date)

def calculate_eia_monthly_consumption_constraints(model,
                                                  eia_daily_observations,
                                                  eia_monthly_data,
                                                  consumption_factor_to_index,
                                                  state: str,
                                                  eia_monthly_start_date="2022-01-01",
                                                  eia_monthly_end_date="2024-01-01",
                                                  sigma=10,
                                                  app_params: dict = None):
    """
    Calculate eia monthly consumption constraints.
    """

    logger = app_params["log_handler"]
    file_handler = app_params["file_handler"]

    logger.info("Calculating EIA Monthly Constraints...")

    eia_monthly_start_date = datetime.datetime.strptime(eia_monthly_start_date, "%Y-%m-%d")
    eia_monthly_end_date = datetime.datetime.strptime(eia_monthly_end_date, "%Y-%m-%d")
    eia_monthly_data["Date"] = pd.to_datetime(
        eia_monthly_data.index.to_series().apply(lambda x: datetime.datetime.strptime(x + "-01","%Y-%m-%d")))

    constraint_random_variables = dict()
    for index, row in eia_monthly_data.iterrows():
        start_month_dt = row["Date"]
        start_date_str = start_month_dt.strftime("%Y-%m-%d")
        year = start_month_dt.year
        month = start_month_dt.month
        day = start_month_dt.day

        if is_data_between_dates(eia_monthly_start_date, eia_monthly_end_date, start_month_dt):

            logger.info(f"Applying Constraint for {start_date_str} with index {index}")

            monthly_value = float(row[state])

            if ((monthly_value is float("nan"))
                or monthly_value is float("inf")
                or (monthly_value is float and np.isclose(monthly_value, 0))):

                continue

            day_of_week, end_of_month_day_number = calendar.monthrange(year, month)
            end_of_month_datetime = datetime.datetime(year, month, end_of_month_day_number)
            indicies = []
            for date in consumption_factor_to_index:
                if (start_month_dt <= date) and (date <= end_of_month_datetime):
                    indicies.append(consumption_factor_to_index[date])

            constraint_random_variable = pm.Normal(f"month_{start_date_str}",
                                                   mu=sum([eia_daily_observations[index] for index in indicies]),
                                                   sigma=sigma,
                                                   observed=monthly_value)

            constraint_random_variables[start_date_str] = constraint_random_variable

    file_handler.flush()
    return constraint_random_variables

def save_parameters(accuracy_result):
    pickle.dump(accuracy_result, open("accuracy_result.pkl", "wb"))


def get_population(state: str):

    if state == "Virginia":
        return VirginiaPopulationData()
    else:
        raise NotImplementedError("State {state} not implemented.".format(state=state))

def get_eia_residential_data(start_date: datetime.date, end_date: datetime.date):
    """
    Gets the EIA Residential data between the start_date and end_date.

    :return:
    """

    residential_df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                              end_date=end_date,
                                                              canonical_component_name="Residential")


    return residential_df


def calculate_eia_data(eia_data: pd.DataFrame, state):
    """
    Calculates the EIA data.


    :param eia_data:
    :return:
    """


    eia_data = eia_data.reset_index()
    eia_data.index = eia_data["period"]
    period = eia_data["period"].apply(lambda x: x + "-01")
    eia_data[state] = eia_data[state].astype(float)
    eia_data["Date"] = pd.to_datetime(period)
    eia_data["Month"] = eia_data["Date"].dt.month
    eia_data["Year"] = eia_data["Date"].dt.year
    eia_data["Day"] = eia_data["Date"].dt.day
    month_to_eia_average = eia_data[["Month", state]].groupby(["Month"])[state].mean().to_dict()
    eia_data["month_average"] = eia_data["Month"].apply(lambda x: month_to_eia_average[x])
    eia_data["month_diff"] = eia_data[state] - eia_data["month_average"]
    return eia_data


def calculate_eia_average_daily_values(eia_data, consumption_factor):

    month_to_eia_average = eia_data[["Month", "month_average"]].groupby(["Month"]).mean().to_dict()["month_average"]
    consumption_factor["Month"] = consumption_factor["Date"].dt.month
    consumption_factor["Year"] = consumption_factor["Date"].dt.year
    consumption_factor["Day"] = consumption_factor["Date"].dt.day
    day_to_average_weather = consumption_factor[["Date", "avg_dd", "Year", "Month", "Day"]]
    day_to_average_weather = day_to_average_weather.groupby(["Month"])["avg_dd"].sum().to_dict()
    hdd_to_eia_per_month = dict()
    for month in month_to_eia_average:
        month_hdd = day_to_average_weather[month]
        month_eia = month_to_eia_average[month]
        hdd_to_eia_per_month[month] = month_eia / month_hdd
    consumption_factor["eia_daily_average"] = consumption_factor.apply(lambda row: row["avg_dd"] * hdd_to_eia_per_month[row["Month"]], axis=1)
    eia_daily_average = consumption_factor[["Date", "eia_daily_average"]]
    return eia_daily_average

def load_residential_data(state,
                          start_training_time,
                          end_training_time,
                          consumption_factor_method="POPULATION_WEIGHTED_HDD",
                          differencing=False,
                          app_params=None):
    """
    Loads residential related data primarily from EIA into a dictionary.

    The function also does preprocessing of the residential data.

    :return:
    """


    #TODO: Handle the future date.
    #if end_training_time > datetime.datetime.now():
    #    raise ValueError("End Training Time cannot be in the future.")

    file_handler = app_params["file_handler"]
    log_handler = app_params["log_handler"]

    log_handler.info("Acquiring EIA Residential Data")
    eia_data = get_eia_residential_data(start_training_time, end_training_time)

    eia_data = eia_data[[state]]
    log_handler.info(f"Finished EIA Residential Data. Some EIA Data is provided as: {eia_data.head()}")

    if consumption_factor_method == "POPULATION_WEIGHTED_HDD":
        population_weighted_weather = PrescientWeather([state])
        consumption_factor = calculate_consumption_factor_via_pop_weighted_weather(population_weighted_weather,
                                                                                  start_training_time,
                                                                                  end_training_time,
                                                                                  state,
                                                                                  differencing=differencing)


        eia_data = calculate_eia_data(eia_data, state)
        if differencing:
            eia_data[state] = eia_data["month_diff"]

    elif consumption_factor_method == "CUSTOM_WITH_PYWEATHER":

        population = get_population(state)
        weather_service = PyWeatherData(population)
        consumption_factor = calculate_consumption_factor(population,
                                                          weather_service,
                                                          start_training_time,
                                                          end_training_time)
    else:
        consumption_factor = None
        eia_data = None

    data = dict()
    data["eia_monthly_values"] = eia_data
    data["full_eia_data"] = eia_data
    data["consumption_factor_values"] = consumption_factor
    data["state"] = state
    data["eia_average_daily_values"] = calculate_eia_average_daily_values(eia_data, consumption_factor)

    return data, consumption_factor, eia_data


def fit_residential_model(start_training_time: str,
                          end_training_time: str,
                          eia_start_time: str,
                          eia_end_time: str,
                          state: str,
                          method="GLOBAL",
                          consumption_factor_method="POPULATION_WEIGHTED_HDD",
                          differencing=False,
                          app_params=None):
    """
    Fits the residential model.

    https://en.wikipedia.org/wiki/List_of_cities_and_counties_in_Virginia

    :return:
    """

    log_handler = app_params["log_handler"]
    file_handler = app_params["file_handler"]

    log_handler.info("Begin fitting residential model...")
    file_handler.flush()

    data, consumption_factor, eia_data = load_residential_data(state,
                                                               start_training_time,
                                                               end_training_time,
                                                               consumption_factor_method=consumption_factor_method,
                                                               app_params=app_params,
                                                               differencing=differencing)


    calibrated_parameters = calibration(consumption_factor,
                                        eia_data,
                                        state)

    calibrated_parameters["daily_consumption_error"] = 0.03

    params = dict()
    params["alpha_mu"] = 0.0
    params["alpha_2_mu"] = 0.0
    params["alpha_sigma"] = 0.0
    params["daily_consumption_error"] = 0.0

    if method == "GLOBAL":
        best_parameters, optimal_rel_error = ResidentialModel(calibrated_parameters,
                                                              params).run_inference_engine_with_global_optimization(
                                                                                                                    start_training_time,
                                                                                                                    end_training_time,
                                                                                                                    eia_start_time,
                                                                                                                    eia_end_time,
                                                                                                                    data,
                                                                                                                    app_params=app_params)

    elif method == "LINEAR":

        best_parameters, optimal_rel_error = ResidentialModelLinearRegression(calibrated_parameters, params).run_inference_engine(start_training_time,
                                                                                                                                end_training_time,
                                                                                                                                eia_start_time,
                                                                                                                                eia_end_time,
                                                                                                                                params,
                                                                                                                                data,
                                                                                                                                app_params)

    else:
        best_parameters, optimal_rel_error = ResidentialModel(calibrated_parameters, params).run_inference_engine(
                                                                                                                start_training_time,
                                                                                                                end_training_time,
                                                                                                                eia_start_time,
                                                                                                                eia_end_time,
                                                                                                                params,
                                                                                                                data,
                                                                                                                app_params)


    log_handler.info("Parameters are provided by {params} ".format(params=best_parameters))
    log_handler.info(f"Relative Error {optimal_rel_error}".format(val=optimal_rel_error))
    file_handler.flush()

if __name__ == '__main__':
    pass



