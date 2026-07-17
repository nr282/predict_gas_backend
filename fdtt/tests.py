"""
Tests that are used to verify the performance of the variational framework.


"""

import unittest
import coverage
import os
from variational_framework import (create_and_solve_problem_1,
                                   indicator, h, g,
                                   integral_of_g,
                                   create_and_solve_problem_2,
                                   create_and_solve_problem_3,
                                   create_and_solve_problem_4,
                                   create_and_solve_problem_0,
                                   create_and_solve_problem_5)
from pandas_add_on import (create_column_dictionaries,
                           solve_dataframe,
                           solve_pandas_series
                           )
import numpy as np
from scipy import integrate
import pandas as pd

class TestVariationalFramework(unittest.TestCase):


    def test_problem_0_variational_framework(self):

        problem_0_solved = create_and_solve_problem_0()
        self.assertTrue(problem_0_solved)


    def test_problem_1_variational_framework(self):
        problem_1_solved = create_and_solve_problem_1()
        self.assertTrue(problem_1_solved)


    def test_problem_2_variational_framework(self):
        problem_2_solved = create_and_solve_problem_2()
        self.assertTrue(problem_2_solved)


    def test_problem_3_variational_framework(self):
        problem_2_solved = create_and_solve_problem_3()
        self.assertTrue(problem_2_solved)


    def test_problem_4_variational_framework(self):
        """
        Problem 3 can be solved using the variational framework.

        It currently fails significantly.

        This has been fixed the problem was the location of the
        integration of the linear term and constant term.

        """


        problem_4_solved = create_and_solve_problem_4()
        self.assertTrue(problem_4_solved)

    def test_indicator_function(self):

        x_start = 10
        x_end = 20
        x = 15
        value = indicator(x, x_start, x_end)

        self.assertTrue(np.isclose(value, 1.0))



    def test_h_x(self):


        x_start = 10
        x_end = 20
        x = 15
        value = h(x, x_start, x_end)

        self.assertTrue(np.isclose(value, 5.0))



    def test_1_h_x(self):


        x_start = 10
        x_end = 20
        x = 5
        value = h(x, x_start, x_end)

        self.assertTrue(np.isclose(value, 0.0))


    def test_2_h_x(self):


        x_start = 10
        x_end = 20
        x = 20
        value = h(x, x_start, x_end)
        actual_value = 10

        self.assertTrue(np.isclose(value,actual_value))



    def test_3_h_x(self):
        x_start = 10
        x_end = 20
        x = 30
        value = h(x, x_start, x_end)
        actual_value = 10

        self.assertTrue(np.isclose(value, actual_value))


    def test_2_h_x(self):
        x_start = 10
        x_end = 20
        x = 0
        value = h(x, x_start, x_end)
        actual_value = 0

        self.assertTrue(np.isclose(value, actual_value))

    #What is important here. We need to look at something like
    #looking at h(x).

    def test_3_g_x(self):
        x_start = 10
        x_end = 20
        x = 30
        value = g(x, x_start, x_end)
        actual_value = 1/2.0 * pow(20 - 10, 2) + 10 * (30 - 20)

        self.assertTrue(np.isclose(value, actual_value))


    def test_1_derivative_of_g_is_h(self):

        x_start = 10
        x_end = 20
        x = 30
        h_step = 0.01
        derivative_g = (g(x+h_step, x_start, x_end) - g(x, x_start, x_end)) / h_step
        h_value = h(x, x_start, x_end)
        self.assertTrue(np.isclose(h_value, derivative_g))

    def test_2_derivative_of_g_is_h(self):

        x_start = 10
        x_end = 20
        x = 15
        h_step = 0.0001
        derivative_g = (g(x+h_step, x_start, x_end) - g(x, x_start, x_end)) / h_step
        h_value = h(x, x_start, x_end)
        self.assertTrue(np.isclose(h_value, derivative_g))


    def test_3_derivative_of_g_is_h(self):

        x_start = 10
        x_end = 20
        x = 0
        h_step = 0.0001
        derivative_g = (g(x+h_step, x_start, x_end) - g(x, x_start, x_end)) / h_step
        h_value = h(x, x_start, x_end)
        self.assertTrue(np.isclose(h_value, derivative_g))


    def test_3_g_x(self):
        x_start = 10
        x_end = 20
        x = 15
        value = g(x, x_start, x_end)
        actual_value = 1/2.0 * pow(15 - 10, 2)

        self.assertTrue(np.isclose(value, actual_value))



    def test_1_of_integral_of_g(self):


        x_start = 10
        x_end = 20

        a = 6
        b = 25

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)
        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_2_of_integral_of_g(self):

        x_start = 5
        x_end = 20

        a = 6
        b = 25

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)

        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_3_of_integral_of_g(self):

        x_start = 5
        x_end = 20

        a = 15
        b = 25

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)

        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))



    def test_4_of_integral_of_g(self):

        x_start = 5
        x_end = 20

        a = 25
        b = 30

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)

        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_5_of_integral_of_g(self):

        x_start = 0
        x_end = 10

        a = 25
        b = 30

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)

        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_6_of_integral_of_g(self):

        x_start = 0
        x_end = 10

        a = 55
        b = 60

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)

        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_7_of_integral_of_g(self):


        x_start = 5
        x_end = 10

        a = 3
        b = 4

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)



        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))
        self.assertTrue(np.isclose(0.0, numerical_integral_of_g))



    def test_9_of_integral_of_g(self):

        x_start = 5
        x_end = 10

        a = 0.1
        b = 2

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)
        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))
        self.assertTrue(np.isclose(numerical_integral_of_g, 0.0))

    def test_10_of_integral_of_g(self):

        x_start = 5
        x_end = 10

        a = 7
        b = 12

        numerical_integral_of_g, _ = integrate.quad(lambda x: g(x, x_start, x_end), a, b)
        analytical_integral_of_g = integral_of_g(x_start, x_end, a, b)
        self.assertTrue(np.isclose(numerical_integral_of_g, analytical_integral_of_g))


    def test_1_create_column_dictionaries(self):
        """
        Tests create_column_dictionaries function.

        :return:
        """


        column_mapping = dict()
        column_mapping["column_1"] = {"column_1"}
        column_mapping["column_2"] = {"column_2"}
        column_mapping["column_3"] = {"column_3"}
        column_mapping["column_4"] = {"column_1", "column_2"}
        column_mapping["column_5"] = {"column_1", "column_3"}

        calculated_column_name_to_function_id, calculated_function_column_names, calculated_aggregate_column_names = create_column_dictionaries(column_mapping)

        actual_aggregate_columns = set(["column_4", "column_5"])
        aggregate_column_diff = calculated_aggregate_column_names.symmetric_difference(actual_aggregate_columns)

        actual_singular_columns = set(["column_1", "column_2", "column_3"])
        singular_column_diff = calculated_function_column_names.symmetric_difference(actual_singular_columns)

        self.assertTrue(all([column in calculated_column_name_to_function_id for column in ["column_1",
                                                                                           "column_2",
                                                                                           "column_3",
                                                                                           "column_4",
                                                                                           "column_5"]]))



        self.assertTrue(len(calculated_column_name_to_function_id["column_4"].symmetric_difference({1,2})) == 0)
        self.assertTrue(len(calculated_column_name_to_function_id["column_5"].symmetric_difference({1, 3})) == 0)
        self.assertTrue(len(aggregate_column_diff) == 0)
        self.assertTrue(len(singular_column_diff) == 0)



    def test_2_create_column_dictionaries(self):
        """
        Tests create_column_dictionaries function.

        create_column_dictionaries aims to take in the column
        representation of various functions; these functions can be the
        summation of other functions, or just singular functions.

        :return:
        """

        column_mapping = dict()

        (calculated_column_name_to_function_id,
         calculated_function_column_names,
         calculated_aggregate_column_names) = create_column_dictionaries(column_mapping)

        actual_aggregate_columns = set([])
        aggregate_column_diff = calculated_aggregate_column_names.symmetric_difference(actual_aggregate_columns)

        actual_singular_columns = set([])
        singular_column_diff = calculated_function_column_names.symmetric_difference(actual_singular_columns)

        self.assertTrue(len(aggregate_column_diff) == 0)
        self.assertTrue(len(singular_column_diff) == 0)



    def test_3_create_column_dictionaries(self):
        """
        Tests create_column_dictionaries function on a condition
        where a particular aggregate column also contains an
        aggregate.

        :return:
        """

        column_mapping = dict()
        column_mapping["column_1"] = {"column_1"}
        column_mapping["column_2"] = {"column_2"}
        column_mapping["column_3"] = {"column_3"}
        column_mapping["column_4"] = {"column_1", "column_2"}
        column_mapping["column_5"] = {"column_1", "column_4"}

        error_string = None
        try:
            calculated_column_name_to_function_id, calculated_function_column_names, calculated_aggregate_column_names = create_column_dictionaries(
                column_mapping)
        except Exception as e:
            error_string = str(e)


        if error_string is not None:
            self.assertTrue("column_5" in error_string)
        else:
            self.assertTrue(False)


    def test_4_create_column_dictionaries(self):
        """
        Tests create_column_dictionaries function on a condition
        where a particular column contains no columns.

        :return:
        """

        column_mapping = dict()
        column_mapping["column_1"] = {}
        column_mapping["column_2"] = {"column_2"}
        column_mapping["column_3"] = {"column_3"}
        column_mapping["column_4"] = {"column_1", "column_2"}
        column_mapping["column_5"] = {"column_1", "column_4"}

        error_handled = False
        try:
            calculated_column_name_to_function_id, calculated_function_column_names, calculated_aggregate_column_names = create_column_dictionaries(column_mapping)
        except:
            error_handled = True


        self.assertTrue(error_handled)

    def test_generic_solve_over_constrained_dataframe(self):
        """
        Tests the generic solve dataframe on an over-constrained
        dataframe.

        The dataframe is over-constrained because, for instance, for
        the date of "2024-01-01", test_state_1 value is 300, test_state_2 value is 300,
        and test_country is 300. test_country's value is the summation of (1) test_state_1
        and (2) test_state_2, which when added together is 600, which is not the value provided
        by test_country.

        :return:
        """




        df = pd.DataFrame.from_dict({"Date": ["2024-01-01",
                                              "2024-02-01",
                                              "2024-03-01",
                                              "2024-04-01",
                                              "2024-05-01",
                                              "2024-06-01",
                                              "2024-07-01",
                                              "2024-08-01",
                                              "2024-09-01",
                                              "2024-10-01",
                                              "2024-11-01",
                                              "2024-12-01"],
                                     "test_state_1": [300,
                                                      400,
                                                      float('nan'),
                                                      200,
                                                      100,
                                                      100,
                                                      200,
                                                      900,
                                                      1000,
                                                      1100,
                                                      900,
                                                      300],
                                     "test_state_2": [300,
                                                      400,
                                                      float('nan'),
                                                      200,
                                                      100,
                                                      100,
                                                      200,
                                                      float('nan'),
                                                      1000,
                                                      1100,
                                                      float('nan'),
                                                      300],
                                     "test_country": [300,
                                                      400,
                                                      float('nan'),
                                                      200,
                                                      100,
                                                      100,
                                                      200,
                                                      900,
                                                      1000,
                                                      float('nan'),
                                                      900,
                                                      float('nan')]})

        column_mapping = dict()
        column_mapping['test_state_1'] = {"test_state_1"}
        column_mapping['test_state_2'] = {"test_state_2"}
        column_mapping['test_country'] = {"test_state_1", "test_state_2"}

        error_message = None
        try:
            variational_framework = solve_dataframe(df, column_mapping)
        except Exception as e:
            error_message = str(e)

        self.assertTrue("Solve has failed." in error_message)



    def test_generic_solve_dataframe(self):
        """
        Tests the generic solve dataframe on a correctly
        constrained dataframe.

        The goal here is to ensure that we interpolate somewhat accurate
        values on this specific test case.

        :return:
        """

        df = pd.DataFrame.from_dict({"Date": ["2024-01-01",
                                              "2024-02-01",
                                              "2024-03-01",
                                              "2024-04-01",
                                              "2024-05-01",
                                              "2024-06-01",
                                              "2024-07-01",
                                              "2024-08-01",
                                              "2024-09-01",
                                              "2024-10-01",
                                              "2024-11-01",
                                              "2024-12-01"],
                                     "test_state_1": [300,
                                                      400,
                                                      300,
                                                      200,
                                                      100,
                                                      100,
                                                      200,
                                                      900,
                                                      1000,
                                                      1100,
                                                      900,
                                                      300],
                                     "test_state_2": [300,
                                                      400,
                                                      300,
                                                      200,
                                                      100,
                                                      100,
                                                      200,
                                                      float('nan'),
                                                      1000,
                                                      1100,
                                                      float('nan'),
                                                      300],
                                     "test_country": [float('nan'),
                                                      float('nan'),
                                                      float('nan'),
                                                      float('nan'),
                                                      float('nan'),
                                                      float('nan'),
                                                      float('nan'),
                                                      1900,
                                                      float('nan'),
                                                      float('nan'),
                                                      1500,
                                                      float('nan')]})

        column_mapping = dict()
        column_mapping['test_state_1'] = {"test_state_1"}
        column_mapping['test_state_2'] = {"test_state_2"}
        column_mapping['test_country'] = {"test_state_1", "test_state_2"}

        variational_framework, interpolated_df, mapping = solve_dataframe(df, column_mapping)
        solve_successful = variational_framework.is_solve_successful()

        self.assertTrue(solve_successful)
        self.assertTrue(len(interpolated_df) > 0)
        self.assertTrue(len(interpolated_df.dropna()) == len(interpolated_df))