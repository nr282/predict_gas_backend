"""
Pandas is a python library that allows one to work with data. There are a number of
different techniques for interpolation provided in Pandas. Most of these techniques
are based on the interpolation of particular points in the dataframe.The goal here will be
to interpolate the EIA Data. Not only will we look to interpolate EIA Data,
but we will look to interpolate other datasets.



An example dataframe is below:
-------------------------------------------------------------------------------------------


    The different states get assigned to particular functions, while the US Aggregation
    represents a set of functions that are summed and integrate to a particular value.

    Date    State 1     State 2     State 3     State 4     ...     US Aggregation
    ------------------------------------------------------------------------------



Another example dataframe that we will need to handle is provided by:
--------------------------------------------------------------------------------------------

    The different states get assigned to particular functions, while the US Aggregation
    represents a set of functions that are summed and integrate to a particular value.

    Date    US Natural Gas
    ----------------------

We need to be able to map both dataframes into the integral constraints that we setup previously.

We can then solve both problems at the same time.

The core time-scale that will be considered are days. In the future it may be different, in order to make the
code base scale, it may be useful to consider sub period scale and period scale.

Information on packaging python libraries and installing them on PyPI can be found here:
    1. https://packaging.python.org/en/latest/tutorials/packaging-projects/

TODO: We can place this into a relevant API, and look to sell.
TODO: This will allows us to move this into production. There are not methods specifically
TODO: tied to integral constraints. Hence, this will need to be refactored.

TODO: Layer is provided below:
TODO: https://jainsaket-1994.medium.com/creating-a-lambda-function-layer-in-python-without-using-ec2-using-cloudshell-399614a4e3ce

Need to look at things correctly.

"""

#s3_bucket_lambda_code_upload

try:
    import pandas as pd
    import numpy as np
    from data.eia_consumption.variational_framework import VariationalFramework, IntegralConstraintElement, IntegralConstraint
    from datetime import datetime, timedelta
    import calendar
    import math
    import logging
except:
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import calendar
    import math
    import logging



logging.basicConfig(level=logging.INFO)


def _validation(df: pd.DataFrame,
               column_mapping: dict):
    """
    In the following function, one looks at performing validation.

    Validation will include checking that elements in the column mapping
    are in the dataframe.

    Validation will include that every column in dataframe is represented
    in the mapping.

    This method will throw an error if there is an error in the inputs.

    It will return True, if validation passes.

    :return:
    """

    columns_in_dataframe = set(list(df.columns))
    if "Date" in columns_in_dataframe:
        columns_in_dataframe.remove("Date")

    for column_in_mapping in column_mapping:
        if column_in_mapping in columns_in_dataframe:
            pass
        else:
            raise ValueError(f"Column in Column Mapping Does Not Exist in the passed in "
                             f"dataframe. Missing column is: {column_in_mapping}")

    for column_in_mapping in column_mapping:
        columns_tied_to_column_in_mapping = column_mapping.get(column_in_mapping)
        for set_column in columns_tied_to_column_in_mapping:
            if not set_column in columns_in_dataframe:
                raise ValueError(f"Column {set_column} in set associated with base column {column_in_mapping} "
                                 f"does not exist in dataframe. ")

    for column_in_df in columns_in_dataframe:
        if not column_in_df in column_mapping:
            raise ValueError(f"Dataframe column {column_in_df} not found in the column mapping. ")


    return True



def create_column_dictionaries(column_mapping: dict):
    """
    Creates the column


    :param df:
    :param column_mapping:
    :return:
    """


    # Develop mappings between column names and function numbers.
    column_name_to_function_id = dict()
    function_column_names = set()
    aggregate_column_names = set()
    function_id = 1
    for column_in_mapping in column_mapping:
        columns_assigned_to_column_in_mapping = column_mapping.get(column_in_mapping)
        if len(columns_assigned_to_column_in_mapping) == 1:
            function_column_names.add(column_in_mapping)
            column_name_to_function_id[column_in_mapping] = {function_id}
            function_id += 1
        else:
            aggregate_column_names.add(column_in_mapping)

    for column_in_mapping in column_mapping:
        columns_assigned_to_column_in_mapping = column_mapping.get(column_in_mapping)
        if len(columns_assigned_to_column_in_mapping) == 1:
            pass
        elif len(columns_assigned_to_column_in_mapping) == 0:
            raise RuntimeError("Column cannot have no columns assigned to its group. ")
        else:
            for column_assigned_to_column_in_mapping in columns_assigned_to_column_in_mapping:
                function_ids = column_name_to_function_id.get(column_in_mapping, set())
                if column_assigned_to_column_in_mapping in column_name_to_function_id:
                    function_id_set = column_name_to_function_id.get(column_assigned_to_column_in_mapping)
                    if len(function_id_set) == 1:
                        function_id = list(function_id_set)[0]
                        function_ids.add(function_id)
                        column_name_to_function_id[column_in_mapping] = function_ids
                    else:
                        raise RuntimeError(f"Column assigned to column set mapping is an aggregate column. The set "
                                           f"of columns tied to a specific column name have to be columns representing"
                                           f"functions. Offending column name is {column_in_mapping}")
                else:
                    raise RuntimeError("Column assigned to column set mapping is not an aggregate column.")



    return (column_name_to_function_id,
            function_column_names,
            aggregate_column_names)

def get_end_of_month(dt: datetime):
    """
    Get the end of the month for a specific datetime.

    """

    day, num_day = calendar.monthrange(dt.year, dt.month)
    end_of_month = datetime(dt.year, dt.month, num_day)
    return end_of_month

def get_beginning_of_week(dt: datetime):
    """
    Get the beginning of the week for a specific datetime. The beginning of the
    week is the current day minus 7 days.

    This is mostly to fill in missing data arising from a shift in
    the data.

    """

    beginning_of_week = dt - timedelta(days=7)
    return beginning_of_week


def add_integral_constraint():
    """


    :return:
    """

    pass


def cure_dataframe(variational_framework: VariationalFramework,
                   raw_input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cure dataframe by filling in the missing datapoints in the raw input dataframe.

    Return: pd.DataFrame
    """

    pass

def convert_element_to_numeric(value):

    if value is None:
        return float('nan')
    elif isinstance(value, str):
        return float(value)
    elif isinstance(value, float):
        return value
    elif isinstance(value, int):
        return float(value)
    else:
        raise RuntimeError(f"Value {value} cannot be converted to numeric.")


def get_beginning_of_quarter(dt):
    """
    Get the beginning of the quarter for the date
    representing a particular date (albeit usually the end)
    of the quarter.

    For instance, if I am provided with 2019-12-31, then the beginning of the
    quarter is: 2019-09-30.


    :param dt:
    :return:
    """

    import datetime
    #dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d")
    day = dt.day
    month = dt.month
    year = dt.year
    result_date_str = None
    if month in [6, 9, 12]:
        if month == 6:
            result_date_str = "-".join([str(year), "03", "31"])
        elif month == 9:
            result_date_str = "-".join([str(year), "06", "30"])
        else:
            result_date_str = "-".join([str(year), "09", "30"])
    else:
        result_date_str = "-".join([str(year - 1), "12", "31"])

    result_dt = datetime.datetime.strptime(result_date_str, "%Y-%m-%d")
    return result_dt



def solve_dataframe(df: pd.DataFrame,
                    column_mapping: dict,
                    date_is_beginning=True) -> VariationalFramework:
    """
    An incoming df dataframe specifies a set of columns and values. 
    
    Each column represents a function or a set of functions whose integral will 
    take on values during a particular period of time, which will be specified by
    the Date column. 
    
    For instance, the US Aggregate number will state a constraint on the states functions. 
    
    Hence, in the design, not only do we need to pass in the df, but we also need to pass
    in a map between the column and the set of functions that this column makes up. 
    
    Since the functions are themselves made up of columns, we merely pass a mapping that states the columns 
    that a particular column is made up of.
    
    As an example, suppose that there is a country, called test_country that is made up of 2
    states, (1) test_state_1, (2) test_state_2, and (3) test_country. A potential dataframe will then look 
    like:     
    ------------------------------------------------------------------------------------------------------
    
    
    Date        test_state_1        test_state_2        test_country
    
    January     100                 100                 NaN
    February    100                 200                 NaN
    March       NaN                 200                 300
    April       NaN                 200                 400
    
    After taking in this dataframe, there is a set of information that can be derived from the dataset. 
    --------------------------------------------------------------------------------------------------
    
    - test_state_1 is assigned to function 1.
    - test_state_2 is assigned to function 2. 
    - test_country is assigned to the summation of (1) test_state_1 and (2) test_state_2, which
        is the summation of function 1 and function 2. 
        
    In order to allow the function to interpret this information, the column_mapping dictionary
    is passed in.
     
    The question becomes what should this dictionary look like. 
    
    An example mapping dictionary would look like: 
    ----------------------------------------------
    
        column_mapping = {}
        column_mapping["test_state_1"] = {"test_state_1"}
        column_mapping["test_state_2"] = {"test_state_2"}
        column_mapping["test_country"] = {"test_state_1",
                                          "test_state_2}
    

    The incoming "Date" column in df needs to represent the beginning of the time period.

    If it does not represent the beginning of the time period, then things must be fixed
    before using in the future.

    I could also look to take in a parameter that asks the question whether this is the
    beginning or the end of the period.




    :return: 
    """

    #Validate column mapping in context of the dataframe.
    _validation(df, column_mapping)

    assert ("Date" in df)

    #Develop the column name to a set of function ids.
    column_name_to_function_id, function_column_names, aggregate_column_names = create_column_dictionaries(column_mapping)
    number_of_functions = len(function_column_names)
    df['Date'] = pd.to_datetime(df['Date'])
    df["date_is_beginning"] = date_is_beginning
    inferred_frequency = pd.infer_freq(df["Date"])
    df = df.sort_values(by=['Date'], ascending=True)
    data_start_date = df['Date'].min()
    data_end_date = df['Date'].max()

    if inferred_frequency == 'MS':
        if date_is_beginning:
            df["Period_Start_Date"] = df["Date"]
            df["Period_End_Date"] = df["Date"].shift(-1, fill_value = get_end_of_month(data_end_date))
            global_start_date = df["Period_Start_Date"].min()
            global_end_date = df["Period_End_Date"].max()
        else:
            raise NotImplementedError("No implementation for Month Start Frequency and Date represents beginning"
                                      " is False.")
    elif inferred_frequency == "W-FRI":
        if date_is_beginning:
            raise NotImplementedError("No implementation for W-FRI frequency and is beginning is True")
        else:
            df["Period_Start_Date"] = df["Date"].shift(1, fill_value=get_beginning_of_week(data_start_date))
            df["Period_End_Date"] = df["Date"]
            global_start_date = df["Period_Start_Date"].min()
            global_end_date = df["Period_End_Date"].max()

    elif inferred_frequency == "QE-DEC":

        #Develop the relevant information here.
        #Need to implement what we have above.
        #How can we think about this?
        df["Period_Start_Date"] = df["Date"].shift(1, fill_value=get_beginning_of_quarter(data_start_date))
        df["Period_End_Date"] = df["Date"]
        global_start_date = df["Period_Start_Date"].min()
        global_end_date = df["Period_End_Date"].max()
    else:
        raise NotImplementedError(f"Cannot currently handle time frequency provided by the inferred frequency "
                                  f"identifier {inferred_frequency}")


    df["global_start_date"] = global_start_date
    df["global_end_date"] = global_end_date

    df["Days_Between_Global_Start_Date_And_Period_Start_Date"] = df["Period_Start_Date"].apply(lambda dt: (dt - global_start_date).days)
    df["Days_Between_Global_Start_Date_And_Period_End_Date"] = df["Period_End_Date"].apply(lambda dt: (dt - global_start_date).days)

    global_start_time = float(df["Days_Between_Global_Start_Date_And_Period_Start_Date"].min())
    global_end_time = float(df["Days_Between_Global_Start_Date_And_Period_End_Date"].max())

    if global_start_time > global_end_time:
        raise RuntimeError(f"Cannot calculate correct values here. Global Start Time is {global_start_time} and "
                           f"{global_start_date}")

    integral_constraint_id = 1
    integral_constraints = []
    for index, row in df.iterrows():
        start_time = float(row["Days_Between_Global_Start_Date_And_Period_Start_Date"])
        end_time = float(row["Days_Between_Global_Start_Date_And_Period_End_Date"])
        for column_name in row.index:

            column_value = row.get(column_name)

            if column_name in (function_column_names.union(aggregate_column_names)):
                #Look to form and add the integral constraint. The formulation
                #and then addition of the integral constraint will be critical.
                value = convert_element_to_numeric(column_value)

                if not math.isnan(value):
                    if column_name in function_column_names:
                        function_ids = column_name_to_function_id.get(column_name)
                        function_id = list(function_ids)[0]



                        integral_constraint_element = IntegralConstraintElement(function_id,
                                                                              start_time,
                                                                              end_time,
                                                                              integral_constraint_id)

                        integral_constraint_elements = [integral_constraint_element]

                        integral_constraint = IntegralConstraint(integral_constraint_elements,
                                                                 value,
                                                                 integral_constraint_id)



                    elif column_name in aggregate_column_names:
                        function_ids = column_name_to_function_id.get(column_name)
                        integral_constraint_elements = []
                        for function_id in function_ids:
                            integral_constraint_element = IntegralConstraintElement(function_id,
                                                                                      start_time,
                                                                                      end_time,
                                                                                      integral_constraint_id)

                            integral_constraint_elements.append(integral_constraint_element)

                        integral_constraint = IntegralConstraint(integral_constraint_elements,
                                                                   value,
                                                                   integral_constraint_id)

                    else:
                        raise RuntimeError("Column not found")

                    integral_constraint_id += 1
                    integral_constraints.append(integral_constraint)

                else:
                    pass


    logging.info(f"Number of Integral Constraints: {len(integral_constraints)} \n"
                 f"Number of Functions: {number_of_functions} \n"
                 f"Global Start Date: {global_start_time} \n"
                 f"Global End Date: {global_end_time} \n")




    #Apply the integral constraints to the Variational Framework.
    framework = VariationalFramework(integral_constraints,
                                     number_of_functions,
                                     global_start_time,
                                     global_end_time)



    #Solve the dataframe.
    framework.solve(check_integral_result=True)

    raw_input_df = df.copy()
    for index, row in raw_input_df.iterrows():
        for column_name in row.index:
            if column_name in aggregate_column_names.union(function_column_names):
                value = convert_element_to_numeric(row[column_name])
                if math.isnan(value):

                    start_time = row["Days_Between_Global_Start_Date_And_Period_Start_Date"]
                    end_time = row["Days_Between_Global_Start_Date_And_Period_End_Date"]

                    function_ids = column_name_to_function_id.get(column_name)
                    missing_value = 0
                    for function_id in function_ids:
                        missing_value += framework.get_function_integrated(start_time, end_time, function_id)

                    raw_input_df.loc[index, column_name] = missing_value

    corrected_df = raw_input_df

    return framework, corrected_df, column_name_to_function_id


def solve_pandas_series(df: pd.DataFrame,
                        date_is_beginning=True):
    """
    Solves the Pandas Series that will be used in a number of different
    areas including but not limited to the series provided via Natural
    Gas Storage.

    """

    if not "Date" in df:
        raise ValueError("Date column not found in the dataframe.")

    if not "Value" in df:
        raise ValueError("Value column not found in the dataframe.")

    column_mapping = dict()
    column_mapping["Value"] = {"Value"}
    framework, corrected_df, column_name_to_function_id = solve_dataframe(df,
                                              column_mapping=column_mapping,
                                              date_is_beginning=date_is_beginning)
    return framework, corrected_df


































