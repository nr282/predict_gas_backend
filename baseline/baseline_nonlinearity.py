"""
The following aims to be the baseline technique that I think is good enough to commercialize.



"""
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.special import pbdn_seq
from sympy.stats.rv import probability

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
from scipy.optimize import least_squares
from baseline.baseline import ComponentType
from scipy.optimize import minimize
from scipy.optimize import basinhopping


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


def calculate_monthly_eia_values(start_date,
                                 end_date,
                                 consumption_factor_values,
                                 consumption_type: str):

    pass


def calculate_consumption_factor_via_weather(start_date,
                                             end_date,
                                             weather_normal,
                                             weather_values):


    pass

def create_daily_eia_via_weather(eia_monthly_value,
                                start_date,
                                end_date,
                                consumption_factor):
    """
    Given an eia_monthly_value such as accumulated consumption in a particular month
    and given the consumption_factor. On the basis of the consumption factor, form
    the daily values that aggregate up to the eia monthly value.
    """

    pass


def create_weather_values(start_date,
                          end_date,
                          current_date,
                          state,
                          component_type: ComponentType):

    prescient_weather = PrescientWeather([state])
    weather = prescient_weather.get_temperature([state], start_date, end_date, current_date)
    return weather

def create_normal_weather_values(normal_start_date,
                                 normal_end_date,
                                 state,
                                 component_type: ComponentType) -> dict:
    """
    Creates the normal weather values.

    """
    current_date = datetime.datetime.now()
    prescient_weather = PrescientWeather([state])
    weather = prescient_weather.get_temperature([state],
                                                normal_start_date,
                                                normal_end_date,
                                                current_date)

    return weather



def theta_calc_complex(theta: np.ndarray, temp: float) -> float:
    """
    Calculation of the consumption.

    TODO: I continue to like this calculation.
    TODO: It seems to provide not great results.
    TODO: Brute force or a better guess may be required.

    :return:
    """

    ref_temp = theta[0]
    ref_temp_2 = theta[1]
    slope1 = theta[2]
    slope2 = theta[3]



    if temp <= ref_temp:
        if temp <= ref_temp_2:
            return slope1 * (ref_temp - ref_temp_2) + slope2 * (ref_temp_2 - temp)
        else:
            return slope1 * (ref_temp - temp)
    else:
        return 0.0


def theta_calc(theta: np.ndarray, temp: float) -> float:
    """
    Calculation of the consumption.

    TODO: I continue to like this calculation.
    TODO: It seems to provide not great results.
    TODO: Brute force or a better guess may be required.

    :return:
    """

    ref_temp = theta[0]
    slope1 = theta[1]
    if temp <= ref_temp:
        return slope1 * (ref_temp - temp)
    else:
        return 0.0


def plot_theta_calc():

    x = np.linspace(0, 100, 100)
    y = [theta_calc(np.array([65,50,5,10]), x_p) for x_p in x]
    plt.plot(x, y)
    plt.xlabel("Temperature")
    plt.ylabel("Consumption")
    plt.title("Form of Temperature Versus Consumption")
    plt.show()




def create_loss_function(weather_data: dict, eia_monthly_values: pd.DataFrame):

    def loss_function(theta: np.ndarray):
        """
        Maximum Likelihood Estimation for theta. In this function, we look to create
        the maximum likelihood estimate.

        In order to develop the Maximum Likelihood Estimation, we need to first calculate
        the likelihood function.

        :return:
        """

        total_abs_diff = 0
        for i, eia_row in eia_monthly_values.iterrows():
            year = eia_row["Year"]
            month = eia_row["Month"]
            eia_diff = eia_row["diff"]
            dates_in_month = weather_data.get((year, month))
            n = len(dates_in_month)
            s = 0
            for date in dates_in_month:

                normal_average = 0
                normal_values, current_temperature = dates_in_month.get(date)
                num_normal = len(normal_values)
                normal_sum = 0
                for normal_date in normal_values:
                    normal_value = normal_values.get(normal_date)
                    normal_consumption = theta_calc(theta, normal_value)
                    normal_sum += normal_consumption
                normal_average = normal_sum / num_normal
                current_consumption = theta_calc(theta, current_temperature)
                daily_diff = current_consumption - normal_average

                s += daily_diff

            diff = eia_diff - s
            total_abs_diff += abs(diff)

        return total_abs_diff

    return loss_function


def calculate_consumption_factor_to_eia_sensitivity_monthly(eia_start_date,
                                                            eia_end_date,
                                                            eia_monthly_values,
                                                            consumption_factor_values,
                                                            consumption_factor_normal_values,
                                                            state,
                                                            component_type: ComponentType
                                                            ):
    """
    Calculate the consumption factor to eia sensitivity on a month-by-month basis.
    ------------------------------------------------------------------------------
    """

    #TODO: Add preconditions

    eia_start_date_dt = datetime.datetime.strptime(eia_start_date, "%Y-%m-%d")
    eia_end_date_dt = datetime.datetime.strptime(eia_end_date, "%Y-%m-%d")

    eia_monthly_values = eia_monthly_values[eia_monthly_values["Date"] >= eia_start_date_dt]
    eia_monthly_values = eia_monthly_values[eia_monthly_values["Date"] <= eia_end_date_dt]
    weather_monthly_values_data = dict()
    for eia_index, row in eia_monthly_values.iterrows():
        month = row["Month"]
        year = row["Year"]
        consumption_factor_values_month_year = consumption_factor_values.query(f"Month_x == {month} & Year_x == {year}")
        consumption_factor_values_month_year_normal = consumption_factor_normal_values.query(f"Month == {month}")
        weather_dates_to_normal_dates = dict()
        for weather_index, weather_row in consumption_factor_values_month_year.iterrows():
            temperature = weather_row["Temperature"]
            date = weather_row["Date"]
            day = weather_row["Day_y"]
            consumption_normal_values = consumption_factor_values_month_year_normal.query(f"Day == {day}")
            normal_values = dict()
            for normal_index, normal_row in consumption_normal_values.iterrows():
                normal_date = normal_row["Date"]
                normal_temperature = normal_row["Temperature"]
                normal_values[normal_date] = normal_temperature

            weather_dates_to_normal_dates[date] = (normal_values, temperature)

        weather_monthly_values_data[(year, month)] = weather_dates_to_normal_dates

    loss_func = create_loss_function(weather_monthly_values_data, eia_monthly_values)


    theta0 = np.array([30,5])



    #TODO: Adjusting the temperature is critical to avoid
    #TODO: local minimums.
    res = basinhopping(loss_func,
                       theta0,
                       niter=1000,
                       disp=True,
                       stepsize=20,
                       minimizer_kwargs={"method": "L-BFGS-B"},
                       T=10000)

    import logging
    logging.info(f"Theta is provided by basinhopping: {res.x}")

    success = res.success
    theta_result = res.x
    if success == True:
        return theta_result
    else:
        return None


def calculate_eia_normal_monthly_values(normal_start_date,
                                        normal_end_date,
                                        state,
                                        component_type: ComponentType
                                        ) -> dict:
    """
    Calculate monthly eia values based on consumption factor values.

    """


    component_type_to_component_name = {ComponentType.RESIDENTIAL: "Residential",
                                      ComponentType.COMMERCIAL: "Commercial",
                                      ComponentType.ELECTRIC: "Electric"}

    component_name = component_type_to_component_name.get(component_type)

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



def calculate_eia_daily_values_with_nonlinear_fitting(start_date: str,
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
    # Get weather values.
    weather_normal_values = create_weather_values(normal_start_date,
                                                   normal_end_date,
                                                   current_date,
                                                   state,
                                                   component_type)

    assert(type(weather_normal_values) == pd.DataFrame)

    #Get weather values.
    weather_values = create_weather_values(start_date,
                                           end_date,
                                           current_date,
                                           state,
                                           component_type)

    assert(len(weather_values) == len(pd.date_range(start=start_date, end=end_date)))
    assert(not weather_values["Temperature"].isna().any())


    #################################################################
    #Begin EIA Calculation
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
                                                                                weather_values,
                                                                                weather_normal_values,
                                                                                state,
                                                                                component_type)


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

    plot_theta_calc()