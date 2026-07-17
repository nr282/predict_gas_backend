"""
Get EIA Consumption data.

In order to use the API, one needs to also use the relevant api key.

The API Key allows one to use the EIA API and it is required.

There are two API's:
    (1) First API command
    (2) Second API command

Notes:
------

1. Need to consider stochastic behavior of the weather_mod and the eia eia_consumption.
2. Primarily, I think that it will be more interesting to look at the stochastic behavior  of the weather_mod.



"""


import os
import time

import pandas as pd
import requests
import logging
import io
import json

from dateutil.relativedelta import relativedelta

from data.eia_consumption.eia_api import read_eia_path
from data.eia_consumption.eia_geography_mappings import (convert_native_name_to_standard_state_name,
                                                         get_fifty_us_states_and_dc,
                                                         get_united_states_name)
from datetime import datetime
from data.eia_consumption.pandas_add_on import solve_dataframe
import math
from data.eia_consumption.pandas_add_on import solve_pandas_series
from scipy import integrate
import calendar
import matplotlib.pyplot as plt
from data.eia_consumption.global_configurations import working_directory_location
from ..weather import get_weather_data
import boto3
from io import StringIO


def download_dataframe_from_s3_bucket():


    s3 = boto3.resource('s3',
                        aws_access_key_id=get_access_key(),
                        aws_secret_access_key=get_secret_access_key())
    bucketname = get_name_of_s3_bucket()
    filename = get_file_name()

    obj = s3.Object(bucketname, filename)

    start = time.time()
    body = obj.get()['Body'].read()
    end = time.time()
    print(f"Difference in Time is: {end - start}")



    data_str = body.decode('utf-8')
    daily_df = pd.read_csv(StringIO(data_str))
    daily_df_columns = list(daily_df.columns)
    daily_df_columns.remove("Unnamed: 0")
    daily_df = daily_df.filter(items=daily_df_columns)



    return daily_df

def upload_df_to_s3_bucket(df: pd.DataFrame):
    """
    Uploads dataframe to s3 bucket.

    """

    from io import StringIO

    bucket_name = get_name_of_s3_bucket()  # already created on S3
    csv_buffer = StringIO()
    df.to_csv(csv_buffer)
    s3_resource = boto3.resource('s3',
                                 aws_access_key_id=get_access_key(),
                                 aws_secret_access_key=get_secret_access_key()
                                 )
    filename = get_file_name()
    s3_resource.Object(bucket_name, filename).put(Body=csv_buffer.getvalue())


def get_eia_consumption_data():


    df = get_eia_consumption_data_df(create_new_data=True,
                                     start_date="2022-01-01",
                                     end_date="2024-12-31")
    return df

def get_daily_weather_data(start_date, end_date, state):

    start_date_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
    df = get_weather_data(start_date_dt, end_date_dt, number_of_locations=6)
    return df

def get_eia_consumption_file_name(state, eia_start_month, eia_end_month):

    return f"eia_monthly_consumption_{str(state)}_{str(eia_start_month)}_{str(eia_end_month)}.csv"

def get_eia_monthly_consumption(eia_start_month, eia_end_month, state="Virginia", consumption_type="Residential"):

    consumption_file_name = get_eia_consumption_file_name(state, eia_start_month, eia_end_month)
    if os.path.exists(consumption_file_name):
        df = pd.read_csv(consumption_file_name)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        return df

    df = get_eia_consumption_data()
    df["Date"] = df["period"].apply(lambda dt: dt + "-01")
    df["Datetime"] = pd.to_datetime(df["Date"])
    df = df.query(f"standard_state_name == '{state}'")
    df_consumption = df[df["series-description"].apply(lambda s: consumption_type in s)] #TODO: This could be replaced with a filter.
    n, p = df.shape
    if n == 0:
        raise RuntimeError(f"No {state} Data is available.")

    df_consumption = df_consumption[df_consumption["Datetime"] >= datetime.strptime(eia_start_month, "%Y-%m-%d")]
    df_consumption = df_consumption[df_consumption["Datetime"] <= datetime.strptime(eia_end_month, "%Y-%m-%d")]
    df_consumption.to_csv(consumption_file_name)
    return df_consumption

def get_eia_mapping(canonical_component_name: str) -> dict:
    """
    Gets the mapping between the canonical component name and the EIA Component name.

    :param canonical_component_name:
    :return:
    """

    canonical_component_name_to_eia_name = dict()
    canonical_component_name_to_eia_name["Residential"] = "Residential Consumption"
    canonical_component_name_to_eia_name["Commercial"] = "Commercial Consumption"
    canonical_component_name_to_eia_name["Electric"] = "Electric Power Consumption"
    canonical_component_name_to_eia_name["Electric Power Consumption"] = "Electric Power Consumption"

    return canonical_component_name_to_eia_name


def get_api_test_path():
    """
    Gets the API call correctly for a test example.

    """

    return r"""https://api.eia.gov/v2/seriesid/ELEC.SALES.CO-RES.A?api_key=b8443fd367021d8fe4de53869989c0f2"""



def get_eia_consumption_path(start_date: str, end_date: str):
    """
    Get EIA Consumption data.

    It takes some searching to find the correct path to use in api path
    provided below.

    A few things to note:
        1. API Dashboard is useful to development of the correct path.
        2. Adding /data to the path is useful to get all data under a particular
        path.
        3. One needs to add the ?api_key=api_key=b8443fd367021d8fe4de53869989c0f2 at the
        end of the link.
        4. The useful api path is not in the API Dashboard but instead in the browser API in the
        search bar at the top.

    Useful links are provided by:
    -----------------------------
        1. https://www.eia.gov/opendata/documentation.php
        2. https://www.eia.gov/opendata/browser/

    In addition to query the relevant, values and not just what data might be there one needs to
    also add the &data[]=value, which is provided at the bottom of the documentation.

    We have some more information that is provided via the API:
    -----------------------------------------------------------
    -----------------------------------------------------------

    The response is a very large data set. We didn't specify any facets or filters,
    so the API returned as many values as it could. The API will not return more
    than 5,000 rows of data points. However, it will identify the total number of
    rows that are responsive to our request in the response header. In this case,
    7,440 data rows match the API request we just made.

    """

    return r"""https://api.eia.gov/v2/natural-gas/cons/sum/data?api_key=b8443fd367021d8fe4de53869989c0f2&data[]=value&start={0}&end={1}""".format(start_date, end_date)


def read_eia_consumption_data(start_date, end_date):

    eia_consumption_path = get_eia_consumption_path(start_date, end_date)
    result = read_eia_path(eia_consumption_path)
    return result

def test_read_api_path():

    eia_test_path = get_api_test_path()
    result = read_eia_path(eia_test_path)
    return result


def get_path_to_raw_eia():
    return "raw_eia_consumption_data.csv"


def get_next_month_first_day(current_day: str):

    datetime_object = datetime.strptime(current_day, '%Y-%m-%d')
    next_month_day = datetime_object + relativedelta(months=1)
    year = next_month_day.year
    month = next_month_day.month
    day = next_month_day.day
    next_month_first_day = datetime(year, month, 1)
    next_month_first_day_string = next_month_first_day.strftime("%Y-%m-%d")
    return next_month_first_day_string

def check_eia_consumption_result(start_date_str: str,
                                 end_date_str: str,
                                 eia_consumption_df: pd.DataFrame,
                                 state: str = "Virginia"):

    dates = pd.date_range(start_date_str, end_date_str, freq="MS")
    virginia_df = eia_consumption_df[eia_consumption_df["standard_state_name"] == state]
    virginia_periods = set(virginia_df["period"].unique())
    candidate_periods = set([f"{date.year}-{str(date.month).zfill(2)}" for date in dates])
    pct = len(virginia_periods) / len(candidate_periods) * 100.0
    missing_periods = candidate_periods - virginia_periods
    threshold = 95.0
    return pct > threshold


def get_eia_consumption_data_bulk_df_yearly(start_date_str: str,
                                         end_date_str: str,
                                         create_new_data=False):

    start_date_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    start_year = start_date_dt.year
    start_month = start_date_dt.month
    end_year = end_date_dt.year
    end_month = end_date_dt.month

    years = list(range(start_year + 1, end_year))


    if start_year == end_year and start_month <= end_month:
        return get_eia_consumption_data_bulk_df(start_date_str,
                                         end_date_str,
                                         create_new_data=create_new_data)
    elif start_year == end_year-1:
        return get_eia_consumption_data_bulk_df(start_date_str,
                                         end_date_str,
                                         create_new_data=create_new_data
                                        )
    else:
        start_year_end_date = datetime(start_year, 12, 31)
        start_year_end_date_str = start_year_end_date.strftime("%Y-%m-%d")
        start_df = get_eia_consumption_data_bulk_df(start_date_str,
                                            start_year_end_date_str,
                                            create_new_data=True
                                            )

        end_year_start_date = datetime(end_year, 1, 1)
        end_year_start_date_str = end_year_start_date.strftime("%Y-%m-%d")
        end_df = get_eia_consumption_data_bulk_df(end_year_start_date_str,
                                              end_date_str,
                                              create_new_data=True)
        year_dfs = []
        for year in years:
            year_start_date = datetime(year, 1, 1)
            year_start_date_str = year_start_date.strftime("%Y-%m-%d")
            year_end_date = datetime(year, 12, 31)
            year_end_date_str = year_end_date.strftime("%Y-%m-%d")
            df = get_eia_consumption_data_bulk_df(year_start_date_str,
                                                year_end_date_str,
                                                create_new_data=True)
            year_dfs.append(df)

        return pd.concat(year_dfs + [start_df] + [end_df])

def get_eia_consumption_data_bulk_df(start_date_str: str,
                                     end_date_str: str,
                                     create_new_data=False):

    next_month_first_day_str = get_next_month_first_day(end_date_str)

    api_call_successful, result = read_eia_consumption_data(start_date_str,
                                                            next_month_first_day_str)

    urlData = result.content
    urlDataDecoded = urlData.decode('utf-8')
    res = json.loads(urlDataDecoded)
    response = res.get('response')

    total = response.get('total')
    date_format = response.get('dateFormat')
    frequency = response.get('frequency')
    warnings = response.get('warnings')
    data = response.get('data')

    eia_consumption_df = pd.DataFrame.from_records(data)

    if not "area-name" in eia_consumption_df:
        raise ValueError("area-name column is not present in the dataframe")

    eia_consumption_df["standard_state_name"] = eia_consumption_df["area-name"].apply(
        lambda area_name: convert_native_name_to_standard_state_name(area_name))


    eia_consumption_df_good = check_eia_consumption_result(start_date_str,
                                                         end_date_str,
                                                         eia_consumption_df,
                                                         "Virginia")

    if eia_consumption_df_good:
        logging.info(f"EIA Consumption data for {start_date_str} to {end_date_str} is good")
    else:
        logging.critical(f"EIA Consumption data for {start_date_str} to {end_date_str} is bad")

    return eia_consumption_df


def get_eia_consumption_data_df(start_date = "2024-01-01",
                                end_date = "2024-10-01",
                                create_new_data=False):
    """
    Gets EIA Consumption data dataframe.

    The fields of the response are:
        1. 'warnings'
        2. 'total',
        3. 'dateFormat'
        4. 'frequency',
        5. 'data'

    """

    logging.info(f"get eia consumption data dataframe {start_date} to {end_date}")

    if os.path.exists(get_path_to_raw_eia()) and not create_new_data:
        return pd.read_csv(get_path_to_raw_eia())

    interval_range = pd.interval_range(start=datetime.strptime(start_date, "%Y-%m-%d"),
                                       end=datetime.strptime(get_next_month_first_day(end_date), "%Y-%m-%d"),
                                       freq="MS",
                                       closed='left')

    dfs = []
    for interval in interval_range:

        start_date_str = str(interval.left)[:10]
        end_date_str = str(interval.right)[:10]

        logging.info(f"get eia consumption data dataframe {start_date_str} to {end_date_str}")

        api_call_successful, result = read_eia_consumption_data(start_date_str,
                                                                end_date_str)

        urlData = result.content
        urlDataDecoded = urlData.decode('utf-8')
        res = json.loads(urlDataDecoded)
        response = res.get('response')

        total = response.get('total')
        date_format = response.get('dateFormat')
        frequency = response.get('frequency')
        warnings = response.get('warnings')
        data = response.get('data')

        eia_consumption_df = pd.DataFrame.from_records(data)

        if not "period" in eia_consumption_df:
            raise RuntimeError(f"Period not found in the eia_consumption_df for the interval {interval}")

        assert (eia_consumption_df["period"].nunique() == 1)
        dfs.append(eia_consumption_df)

    eia_consumption_df = pd.concat(dfs)

    eia_consumption_df["standard_state_name"] = eia_consumption_df["area-name"].apply(
        lambda area_name: convert_native_name_to_standard_state_name(area_name))

    return eia_consumption_df


def query_eia_consumption_data(eia_consumption_df,
                               canonical_component_name: str):
    """
    Query the dataframe for the particular component.



    """

    eia_native_component_column_name = "process-name"
    eia_native_name = get_eia_mapping(canonical_component_name).get(canonical_component_name)
    eia_consumption_df_for_component_name = (eia_consumption_df[eia_consumption_df[
        eia_native_component_column_name].isin([eia_native_name])])

    return eia_consumption_df_for_component_name

def verify_dataframe_has_target_info(df, start_date, end_date):

    start_period = start_date[:-3]
    end_period = end_date[:-3]
    target_has_info = ("period" in df.columns and
                    start_period in df["period"].astype(str).unique()
                    and end_period in df["period"].astype(str).unique())

    return target_has_info

def get_eia_consumption_data_in_pivot_format(start_date = "2000-01-01",
                                             end_date = "2024-10-01",
                                             canonical_component_name: str = "Residential",
                                             create_new_data=True):
    """
    Get all eia data from 2000-01-01 to 2024-01-01.
    The method calls EIA multiple times to get results.


    

    """

    if not create_new_data:
        cannot_download = False
        df = None
        try:
            df = download_dataframe_from_s3_bucket()
            verified = verify_dataframe_has_target_info(df, start_date, end_date)
            if verified:
                return df
        except:
            cannot_download = True

    eia_consumption_df = get_eia_consumption_data_bulk_df_yearly(start_date,
                                                                end_date,
                                                                create_new_data=create_new_data)

    eia_consumption_df["Units"] = "MCCF"
    eia_consumption_for_component_df = query_eia_consumption_data(eia_consumption_df,
                                    canonical_component_name)


    try:
        upload_df_to_s3_bucket(eia_consumption_for_component_df)
    except Exception as e:
        logging.critical(f"Could not upload dataframe to s3 bucket. The exception is {e}")

    #What are the columns that are required.
    #Look to pivot to get the correct column names.
    #https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pivot.html

    #Clearly, index and columns can repeat but their tuples cannot repeat.
    #Drop duplicates is a good way forward.

    eia_consumption_for_component_df = eia_consumption_for_component_df.groupby(["period", "standard_state_name"])["value"].max().reset_index()
    eia_consumption_for_component_df = eia_consumption_for_component_df.pivot(index="period",
                                                                              columns="standard_state_name",
                                                                                values="value")

    eia_consumption_for_component_df = eia_consumption_for_component_df.reset_index()
    eia_consumption_for_component_df["Date"] = pd.to_datetime(eia_consumption_for_component_df["period"].apply(lambda x: datetime.strptime(x + "-01", "%Y-%m-%d")))
    eia_consumption_for_component_df["Month"] = eia_consumption_for_component_df["Date"].apply(lambda x: x.month)

    assert("Month" in eia_consumption_for_component_df.columns)
    assert("period" in eia_consumption_for_component_df.columns)



    return eia_consumption_for_component_df


def calculate_error_in_df(row: pd.Series):
    """
    Calculate error in the incoming EIA dataframe.


    :param row:
    :return:
    """


    united_states_name = get_united_states_name()
    state_names = get_fifty_us_states_and_dc()
    if united_states_name in row:
        united_states_value = row.get(united_states_name)
        if not type(united_states_value) == float:
            if (type(united_states_value) is str and united_states_value.isnumeric()) or united_states_value is None:
                if united_states_value is None:
                    united_states_value = float('nan')
                else:
                    united_states_value = float(united_states_value)
            else:
                raise ValueError("Value parsed from the dataframe is not. The value is provided "
                                 "by {}".format(united_states_value))
        else:
            pass
    else:
        raise RuntimeError(f"United States Value is not provided")

    united_states_value_aggregated_from_state_level = 0
    state_data_fully_provided = True
    for state_name in state_names:
        if state_name in row:
            state_value = row.get(state_name)
        else:
            raise RuntimeError(f"State provided by {state_name} is not provided")

        if state_value is None or state_value.isnumeric():
            if state_value is None:
                state_value = float('nan')
            else:
                state_value = float(state_value)
        else:
            raise ValueError(f"State value parsed from dataframe is a string not a number")

        if not math.isnan(state_value):
            united_states_value_aggregated_from_state_level += state_value
        else:
            state_data_fully_provided = False
            break

    if state_data_fully_provided:
        return united_states_value - united_states_value_aggregated_from_state_level
    else:
        return None


def calculate_state_aggregated_us_value_in_df(row: pd.Series):
    """
    Calculate state aggregated US value. This is equivalent to
    the summation of all 50 states that make up United States.

    """

    united_states_name = get_united_states_name()
    state_names = get_fifty_us_states_and_dc()
    if united_states_name in row:
        united_states_value = row.get(united_states_name)
        if not type(united_states_value) == float:
            if (type(united_states_value) is str and united_states_value.isnumeric()) or united_states_value is None:
                if united_states_value is None:
                    united_states_value = float('nan')
                else:
                    united_states_value = float(united_states_value)
            else:
                raise ValueError("Value parsed from the dataframe is not. The value is provided "
                                 "by {}".format(united_states_value))
        else:
            pass

    united_states_value_aggregated_from_state_level = 0
    state_data_fully_provided = True
    for state_name in state_names:
        if state_name in row:
            state_value = row.get(state_name)
        else:
            raise RuntimeError(f"State provided by {state_name} is not provided")

        if state_value is None or state_value.isnumeric():
            if state_value is None:
                state_value = float('nan')
            else:
                state_value = float(state_value)
        else:
            raise ValueError(f"State value parsed from dataframe is a string not a number")

        if not math.isnan(state_value):
            united_states_value_aggregated_from_state_level += state_value
        else:
            state_data_fully_provided = False
            break

    if state_data_fully_provided:
        return united_states_value_aggregated_from_state_level
    else:
        return None


def get_name_for_us_error():

    return "error_in_us_state_versus_us_state_aggregate"

def get_state_aggregate_column_name():
    return "us_value_state_aggregate_column_name"

def check_for_data_consistency(eia_consumption_df):
    """
    Runs checks for the EIA Residential pivot df to ensure consistency
    in the problem.

    The checks we would like to run are things like, does the summation of all
    the states add up to the US Aggregate number.

    """


    error_column_name = get_name_for_us_error()
    state_aggregate_column_name = get_state_aggregate_column_name()

    eia_consumption_df[error_column_name] = eia_consumption_df.apply(lambda row: calculate_error_in_df(row),
                                                                                                 axis=1)



    eia_consumption_df[state_aggregate_column_name] = eia_consumption_df.apply(lambda row: calculate_state_aggregated_us_value_in_df(row),
                                                                     axis=1)

    #Need to modify with the state aggregate number.
    #

    inconsistency = (eia_consumption_df[error_column_name].abs().sum() > 0)
    return eia_consumption_df, inconsistency


def apply_variational_framework_on_eia_consumption_data(start_date = "2001-01-01",
                                                        end_date = "2024-10-01",
                                                        fix_inconsistency = True,
                                                        canonical_component_name = "Residential"):
    """
    Apply variational framework on the eia eia_consumption data.



                                Dataframe


            State 1     State 2     State 3     State 4 ... State N
    ------------------------------------------------------------------------------------

    Day 1
    Day 2
    ...
    Day M



    """

    df = get_eia_consumption_data_in_pivot_format(start_date = start_date,
                                                  end_date = end_date,
                                                  canonical_component_name = canonical_component_name)

    column_mapping = dict()
    state_columns = [column for column in df.columns if column in get_fifty_us_states_and_dc()]
    united_states_name = get_united_states_name()
    for state in state_columns:
        column_mapping[state] = {state}

    if df["United States"].isna().sum() > 0:
        raise RuntimeError("Column United States has nans")

    #Fix inconsistency between the state aggregates and the
    #United States if they exist.
    attempted_fix = False
    df, inconsistency_in_aggregates = check_for_data_consistency(df)
    if inconsistency_in_aggregates and fix_inconsistency:
        df["United States"] = df[get_state_aggregate_column_name()].combine_first(df["United States"])
        attempted_fix=True
    else:
        attempted_fix = False

    df, inconsistency_in_aggregates = check_for_data_consistency(df)
    if inconsistency_in_aggregates and fix_inconsistency and attempted_fix:
        raise RuntimeError("Inconsistency found in the dataframe, even though "
                           "a fix was already attempted. ")


    if df["United States"].isna().sum() > 0:
        raise RuntimeError("United States column has NaNs.")

    #Add the two sets together.
    column_mapping[united_states_name] = set(state_columns)
    items = get_fifty_us_states_and_dc() | set([united_states_name])
    df = df.filter(items=items)
    df = df.reset_index()
    df = df.rename(columns={"period": "Date"})
    df['Date'] = df["Date"].apply(lambda x: "-".join([x, "01"]))
    df['Date'] = pd.to_datetime(df['Date'])

    #Save the dataframe to be used in the future.
    #After standardizing the csv file, we can observe
    #it to make sure it will work for the solving of the dataframe
    logging.info("Saving standardizied eia dataframe.")

    framework, corrected_df, column_name_to_function_id = solve_dataframe(df,
                                                                        column_mapping,
                                                                        date_is_beginning=True)

    start_date = corrected_df["Period_Start_Date"].min()
    end_date = corrected_df["Period_End_Date"].max()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    columns_with_us = list(column_name_to_function_id.keys())
    columns_with_us.remove("United States")
    columns_without_us = columns_with_us

    daily_df = pd.DataFrame(index=dates,
                      columns = columns_without_us)

    daily_df = daily_df.reset_index()
    daily_df = daily_df.rename(columns = {"index": "Date"})
    daily_df["Begin_Days_Since_Start_Date"] = daily_df["Date"].apply(lambda dt: (dt - start_date).days)
    daily_df["End_Days_Since_Start_Date"] = daily_df["Begin_Days_Since_Start_Date"] + 1

    for index, row in daily_df.iterrows():
        for column_name in row.index:
            function_id_set = column_name_to_function_id.get(column_name)
            if function_id_set is None:
                pass
            else:
                if len(function_id_set) == 1:
                    function_id = list(function_id_set)[0]
                    start_time = row.get("Begin_Days_Since_Start_Date")
                    end_time = row.get("End_Days_Since_Start_Date")
                    function_lambda = framework.calculate_functions().get(function_id)
                    if function_lambda is None:
                        raise KeyError("Function Id does not exist")
                    else:
                        value, _ = integrate.quad(function_lambda, start_time, end_time)
                        daily_df.loc[index, column_name] = value
                else:
                    pass

    return corrected_df, daily_df

def get_number_of_mmcf_in_bcf():
    return 1000


def get_number_days_in_month(year: int, month: int):


    month_number, number_of_days_in_month = calendar.monthrange(year,month)
    return number_of_days_in_month


def apply_variational_framework_to_us_aggregated_res_comm_consumption_data(start_date = "2001-01-01",
                                                                            end_date = "2024-10-01"):
    """
    Apply the variational framework to US aggregated data.

    The disaggregation of US data is critical.

    """


    residential_df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                              end_date=end_date,
                                                              canonical_component_name="Residential")

    commercial_df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                              end_date=end_date,
                                                              canonical_component_name="Commercial")



    united_states_df = residential_df["United States"].astype('float64').add(commercial_df["United States"].astype('float64'))



    united_states_df = united_states_df.reset_index()
    united_states_df = united_states_df.rename(columns={"period": "Date",
                                                        "United States": "Value"})


    number_of_mmcf_in_bcf = get_number_of_mmcf_in_bcf()

    united_states_df["Date"] = united_states_df["Date"].apply(lambda x: "-".join([x, "01"]))
    united_states_df["Date"] = pd.to_datetime(united_states_df["Date"])
    united_states_df["Value"] = united_states_df["Value"].astype('float32').div(number_of_mmcf_in_bcf)

    if united_states_df["Value"].isna().sum() > 0:
        raise RuntimeError("Column United States has nans")

    framework, corrected_df = solve_pandas_series(united_states_df,
                                                date_is_beginning=True)



    function_id_to_function_lambdas = framework.function_id_to_function_lambdas

    global_start_date = corrected_df["global_start_date"].unique()[0]
    global_end_date = corrected_df["global_end_date"].unique()[0]

    function_lambda = function_id_to_function_lambdas.get(1)



    date_range = pd.date_range(start=global_start_date,
                               end=global_end_date,
                               freq="D")


    date_df = pd.DataFrame(date_range, columns=["Date"])
    date_df["Month_Number"] = date_df["Date"].dt.month
    date_df["Year_Number"] = date_df["Date"].dt.year
    date_df["Number_Of_Days_In_Month"] = date_df.apply(
        lambda row: get_number_days_in_month(row["Year_Number"],
                                             row["Month_Number"]), axis=1)

    date_df["Start_Date"] = global_start_date
    date_df["End_Date"] = global_end_date
    date_df["Days_Since_Start_Date"] = (date_df["Date"] - date_df["Start_Date"]).dt.days
    date_df["Daily_Value"] = date_df["Days_Since_Start_Date"].apply(lambda dt: integrate.quad(function_lambda,
                                                                                              dt,
                                                                                              dt + 1)[0])

    return date_df

def get_number_of_days_in_month(date):
    return calendar.monthrange(date.year, date.month)[1]

def calculate_uniform_disaggregation(united_states_df):
    """
    Calculates the uniform disaggregation in time.

    :return:
    """

    united_states_df["Begin_Date"] = united_states_df["Date"]
    united_states_df["End_Date"] = united_states_df["Date"].shift(-1)

    def daily_uniform_disaggregation(date):

        united_states_filtered = united_states_df[united_states_df["Begin_Date"] <= date]
        united_states_filtered = united_states_filtered[date < united_states_df["End_Date"]]
        if len(united_states_filtered) == 1:
            return float(united_states_filtered["Value"].iloc[0]) / get_number_of_days_in_month(united_states_filtered["Date"].iloc[0])
        else:
            logging.info(f"Cannot calculate uniform disaggregation for the date {date}")
            return None

    return daily_uniform_disaggregation


def apply_variational_framework_to_us_aggregated_electric_power_consumption_data(start_date="2001-01-01",
                                                                                 end_date="2024-10-01"):
    """
    Apply the variational framework to US aggregated data.

    The disaggregation of US data is critical.

    """

    electric_power_df = get_eia_consumption_data_in_pivot_format(start_date=start_date,
                                                              end_date=end_date,
                                                              canonical_component_name="Electric Power Consumption")

    if len(electric_power_df) == 0 or electric_power_df is None:
        raise RuntimeError("Electric Power Dataframe is empty or Electric Power Dataframe is None")

    united_states_df = electric_power_df["United States"].astype('float64')

    united_states_df = united_states_df.reset_index()
    united_states_df = united_states_df.rename(columns={"period": "Date",
                                                        "United States": "Value"})

    number_of_mmcf_in_bcf = get_number_of_mmcf_in_bcf()

    united_states_df["Date"] = united_states_df["Date"].apply(lambda x: "-".join([x, "01"]))
    united_states_df["Date"] = pd.to_datetime(united_states_df["Date"])
    united_states_df["Value"] = united_states_df["Value"].astype('float32').div(number_of_mmcf_in_bcf)

    if united_states_df["Value"].isna().sum() > 0:
        raise RuntimeError("Column United States has nans")

    framework, corrected_df = solve_pandas_series(united_states_df,
                                                  date_is_beginning=True)

    function_id_to_function_lambdas = framework.function_id_to_function_lambdas

    global_start_date = corrected_df["global_start_date"].unique()[0]
    global_end_date = corrected_df["global_end_date"].unique()[0]

    function_lambda = function_id_to_function_lambdas.get(1)

    date_range = pd.date_range(start=global_start_date,
                               end=global_end_date,
                               freq="D")

    date_df = pd.DataFrame(date_range, columns=["Date"])
    date_df["Month_Number"] = date_df["Date"].dt.month
    date_df["Year_Number"] = date_df["Date"].dt.year
    date_df["Number_Of_Days_In_Month"] = date_df.apply(
        lambda row: get_number_days_in_month(row["Year_Number"],
                                             row["Month_Number"]), axis=1)

    date_df["Start_Date"] = global_start_date
    date_df["End_Date"] = global_end_date
    date_df["Days_Since_Start_Date"] = (date_df["Date"] - date_df["Start_Date"]).dt.days
    date_df["FDTT"] = date_df["Days_Since_Start_Date"].apply(lambda dt: integrate.quad(function_lambda,
                                                                                              dt,
                                                                                              dt + 1)[0])

    date_df["Daily_Difference_Value"] = date_df["FDTT"].diff()

    plt.figure(figsize=(10, 10))
    hist = date_df["FDTT"].hist(bins=60)
    plt.title("Histogram of Daily Electric Power Consumption (BCF/DAY)")

    plt.xlabel("Daily (BCF/DAY) In Electric Power Consumption")
    plt.ylabel("Number of Days")
    plt.savefig("daily_electric_power_consumption.png")
    plt.clf()


    plt.figure(figsize=(10, 10))
    hist = date_df["Daily_Difference_Value"].hist(bins=60)
    plt.title("Histogram of Daily Difference in Electric Power Consumption (BCF/DAY)")
    plt.xlabel("Daily Difference (BCF/DAY) In Electric Power Consumption")
    plt.ylabel("Number of Days")
    plt.savefig("daily_difference_electric_power_consumption.png")
    plt.clf()


    #Look to plot the daily values on a graph
    date_df.plot(x="Date", y="FDTT", figsize=(10, 10))
    plt.title("Plot Of Daily Electric Power Consumption Values")
    plt.xlabel("Date")
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.savefig("daily_plot_of_electric_power_consumption.png")
    plt.clf()

    uniform_disaggregation = calculate_uniform_disaggregation(united_states_df)

    date_df["Uniform_Disaggregation"] = date_df["Date"].apply(lambda dt: uniform_disaggregation(dt))
    calculated_data = date_df[["Date", "FDTT", "Uniform_Disaggregation"]].dropna()



    calculated_data.plot(kind='line', x='Date', y=['Uniform_Disaggregation', 'FDTT'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Comparison of FDTT Versus Uniform (Traditional) Disaggregation")
    plt.savefig("comparison_of_uniform_versus_fdtt.png")

    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['Uniform_Disaggregation'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Uniform (Traditional) Disaggregation of EIA Data")
    plt.savefig("traditional_disaggregation.png")

    calculated_data["error"] = (calculated_data["FDTT"] - calculated_data["Uniform_Disaggregation"])
    calculated_data["FDTT_CUM_CONSUMPTION"] = calculated_data["FDTT"].cumsum()
    calculated_data["UNIFORM_CUM_CONSUMPTION"] = calculated_data["Uniform_Disaggregation"].cumsum()

    calculated_data["abs_error"] = (calculated_data["FDTT"] - calculated_data["Uniform_Disaggregation"]).abs()
    calculated_data["pct_error"] = calculated_data["abs_error"].divide(calculated_data["Uniform_Disaggregation"])
    calculated_data["signed_pct_error"] = calculated_data["error"].divide(calculated_data["Uniform_Disaggregation"])
    mean_pct_error = calculated_data["pct_error"].mean()
    max_pct_error = calculated_data["pct_error"].max()

    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['FDTT_CUM_CONSUMPTION', 'UNIFORM_CUM_CONSUMPTION'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Cumulative Consumption")
    plt.savefig("cumulative_consumption_for_fdtt_and_uniform.png")

    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['error'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Difference Of Traditional and Uniform Disaggregation of EIA Data")
    plt.savefig("difference_of_traditional_and_uniform.png")

    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['signed_pct_error'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Percent Error Between Variational and Uniform Disaggregation of EIA Data")
    plt.savefig("signed_percent_error_of_traditional_and_uniform.png")

    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['pct_error'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Percent Error of EIA Data")
    plt.savefig("percent_error_of_traditional_and_uniform.png")

    calculated_data["Consumption_Difference"] = calculated_data["error"].cumsum()

    #TODO: The question here is can we use the error to inform the trading.


    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['Consumption_Difference'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Percent Error of EIA Data")
    plt.savefig("consumption_difference.png")


    plt.clf()
    calculated_data.plot(kind='line', x='Date', y=['Consumption_Difference'], figsize=(8, 8))
    plt.ylabel("Power Consumption (BCF/DAY)")
    plt.title("Consumption Difference of EIA Data")
    plt.savefig("consumption_difference_cumulative_sum.png")


    logging.info("The average percent daily error is provided by: {}".format(mean_pct_error))
    logging.info("The maximum percent daily error is provided by: {}".format(max_pct_error))

    return date_df

def calculate_mean_and_std_for_daily_values_for_consumption(start_date: str,
                                                           end_date: str,
                                                           calculation_years = []):
    """
    Calculates EIA simulated daily values for Residential or Commercial Consumption for a
    particular period of time, beginning with the start_date and ending with the end date.

    Start Date = "2024-03-01"
    End Date = "2024-04-01"
    Calculation Years = [2022, 2023]

    If these are the arguments, then we can look at the (1) mean and (2) standard deviation
    of the values that are found during this period of time.

    The set of dates:
        (1) 2022-03-01 to 2022-04-01
        (2) 2023-03-01 to 2023-04-01

    The mean and standard deviation can be calculated from these set of dates.


    """

    pass



def calculate_random_process_consumption(start_date: str,
                                         end_date: str,
                                         calculation_years = []):
    """
    Calculates a random process for daily values for eia_consumption.

    This will use the (1) calculate_mean_and_std_for_daily_values_for_consumption

    This will be between (1) start_date and (2) end_date

    """


    pass