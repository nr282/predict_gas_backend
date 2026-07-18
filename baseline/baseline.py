"""
The following aims to be the baseline technique that I think is good enough to commercialize.

"""
import matplotlib.pyplot as plt
import numpy as np
import os
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
from utils import get_base_path
import logging



class ComponentType(Enum):
    RESIDENTIAL="RESIDENTIAL"
    COMMERCIAL="COMMERCIAL"
    ELECTRIC="ELECTRIC"

component_to_type = dict()
component_to_type[ComponentType.RESIDENTIAL] = "HDD"
component_to_type[ComponentType.COMMERCIAL] = "HDD"
component_to_type[ComponentType.ELECTRIC] = "CDD"


def calculate_consumption_factor_diff(start_date,
                                      end_date,
                                      normal_weather,
                                      weather_values,
                                      component_type: ComponentType):



    weather_values = calculate_differences_for_df(weather_values, component_to_type[component_type])
    weather_values["diff"] = weather_values.apply(lambda row: row[component_to_type[component_type]]
                                                    - normal_weather[row["day_of_year"]], axis=1)

    weather_values["Day"] = weather_values["Date"].apply(lambda x: x.day)
    weather_values["Year"] = weather_values["Date"].apply(lambda x: x.year)
    weather_values["Month"] = weather_values["Date"].apply(lambda x: x.month)
    return weather_values[["Date", "Year", "Month", "Day", "diff"]]


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


def calculate_consumption_factor_to_eia_sensitivity(start_date,
                                                    end_date,
                                                    eia_monthly_values,
                                                    consumption_factor_values):
    """
    Calculate the eia sensitivity between (a) consumption factor and (b) eia_monthly_value
    between the start_date and end_date.


    """


    pass



def calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                            eia_end_date,
                                                            eia_monthly_values,
                                                            consumption_factor_values,
                                                            state,
                                                            component_type: ComponentType
                                                            ):
    """
    Calculate the consumption factor to eia sensitivity on a month-by-month basis.
    """


    consumption_factor_by_month = consumption_factor_values.groupby(["Year", "Month"])["diff"].sum().reset_index()

    comparison = eia_monthly_values.merge(consumption_factor_by_month,
                                           on=["Year", "Month"],
                                           how="inner", suffixes=("_eia", "_consumption_factor"))

    comparison_with_nan_dropped = comparison.dropna()
    if len(comparison_with_nan_dropped) > 0.8 * len(comparison):
        comparison = comparison_with_nan_dropped
    else:
        raise RuntimeError("Insufficient data to calculate sensitivity. Too many nans")

    working_path = get_base_path()
    if os.path.exists(os.path.join(working_path)):
        try:
            comparison.to_csv(os.path.join(working_path,
                                           "calibration_datasets",
                                           f"{state}_{component_type}_monthly_comparison.csv"))
        except:
            pass

    comparison.plot(x="diff_consumption_factor", y="diff_eia", kind="scatter")
    try:
        plt.savefig(f"{state}_{component_type}_diff.png")
    except:
        pass

    X = comparison["diff_consumption_factor"].values
    y = comparison["diff_eia"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    res = stats.linregress(X_train, y_train)
    y_predict = res.intercept + res.slope * X_test

    error = np.sum(np.abs(y_test - y_predict))
    total_mag = np.sum(np.abs(y_test))

    print(f"Error divided by total mag is {error/total_mag}")

    params = {"slope": res.slope, "intercept": res.intercept}
    return params, 100 * error / total_mag

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
                                            params,
                                            state,
                                            component_type: ComponentType):
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
        date_to_value[date] = daily_value

    result = pd.DataFrame.from_dict({"Date": [key for key in date_to_value],
                                     "Value": [date_to_value[key] for key in date_to_value]})

    result["Date"] = result["Date"].astype(str)
    result["Value"] = result["Value"].astype(float)

    return result


def calculate_non_weather_dependent_component(params,
                                              weather_values,
                                              eia_values,
                                              component_type: ComponentType,
                                              state: str):
    """
    Calculates the time varying minimum consumption for a given component type.

    The natural gas consumption is the addition of:
        (1) time-dependent minimum consumption
        (2) weather-dependent consumption

    It will look to calculate the monthly time series for monthly consumption and will
    look to state what percentage non-weather dependent component is of the total consumption.

    Non-weather dependent consumption is around the 50% level.



    """

    weather_values_by_month = weather_values.groupby(["Year", "Month"])["HDD"].sum().reset_index()
    weather_values_by_month["Consumption"] = weather_values_by_month["HDD"] * params["slope"]
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

    return weather_values_by_month



def calculate_eia_daily_values(start_date: str,
                               end_date: str,
                               eia_start_date: str,
                               eia_end_date: str,
                               normal_start_date: str,
                               normal_end_date: str,
                               current_date: str,
                               component_type: ComponentType,
                               state):
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
    #and consumption factor values.
    params, pct_error = calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                                                eia_end_date,
                                                                                eia_monthly_diff,
                                                                                consumption_factor_diff,
                                                                                 state,
                                                                                 component_type)


    ###############################################################
    #Calculate non-weather dependent component, which is called the
    #time-varying minimum consumption.
    minimum_consumption = calculate_non_weather_dependent_component(params,
                                                                    weather_values,
                                                                    eia_monthly_values,
                                                                    component_type,
                                                                    state)




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
                                                            params,
                                                            state,
                                                            component_type)

    return daily_eia_values, pct_error


if __name__ == "__main__":

    daily_values = calculate_eia_daily_values("2023-01-01",
                               "2025-09-30",
                               "2023-01-01",
                               "2025-09-30",
                               "2019-01-01",
                               "2022-12-31",
                               "2025-09-30",
                               ComponentType.RESIDENTIAL,
                               "Virginia")
