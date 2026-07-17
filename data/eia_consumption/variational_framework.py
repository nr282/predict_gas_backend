"""
In the following code, I will look to build out the relevant variational framework
that will help with a number of trading algorithms and markets. The key element that will be spoken
about is the (1) general variational framework. The general variational framework will be critical
to these problems.

These techniques will widely applied to commodities trading and equities. The core innovation
with these techniques is that we can scale, and calculate quickly various constraints that are
found in traditional datasets.

"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from functools import partial
import scipy.integrate as integrate
import unittest


logging.basicConfig(level=logging.DEBUG)


def indicator(x, x_start, x_end):


    value = None
    if x <= x_start:
        value = 0
    elif x_start <= x and x <= x_end:
        value = 1
    elif x_end < x:
        value = 0

    return value


def h(x, x_start, x_end):

    if x <= x_start:
        return 0
    elif x_start <= x and x <= x_end:
        return x - x_start
    elif x_end < x:
        return x_end - x_start


def g(x, x_start, x_end):

    if x <= x_start:
        return 0
    elif x_start <= x and x <= x_end:
        return 1/2.0 * pow(x - x_start, 2.0)
    else:
        return 1/2.0 * pow(x_end - x_start, 2.0) + (x - x_end) * (x_end - x_start)


def integral_of_g(x_start: float,
                  x_end: float,
                  a: float,
                  b: float):
    """
    Integrating the g(x, x_start, x_end) between a and b.

    When doing this calculation, there are four points:
        (1) x_start
        (2) x_end
        (3) a
        (4) b

    Without loss of generality, one can take two things:
        (1) x_start < x_end
        (2) a < b

    Hence, there are a set of possible combinations that can be looked at.

    + a < b
    + x_start < x_end

    (1) a < b < x_start < x_end
    (2) a < x_start < b < x_end
    (3) a < x_start < x_end < b
    (4) x_start < a < x_end < b
    (5) x_start < a < b < x_end
    (6) x_start < x_end < a < b

    These are the six cases that need to be handled. Below, I look to create the above
    six cases. These six cases are the ones that need to be handled correctly.

    """

    if x_end < x_start or b < a:
        raise ValueError(f'x_end cannot be before x_start and likewise b cannot be before a. '
                         f'These are the conventions '
                         f'that are used in the code. x_start was provided by: '
                         f' {x_start} and x_end was provided by: {x_end} and '
                         f'a was provided by {a}, and b was provided by {b}')

    #Less than or equal needs to be handled properly.
    #What else do we need to think about.
    #When we write inequalities, we need to think about the edge
    #cases in the interval.

    #Need to effectively prove correctness here so to avoid bad
    #edge cases.

    if a <= b and b <= x_start and x_start <= x_end:
        return 0
    elif a < x_start and x_start <= b and b <= x_end:
        return integral_of_g(x_start, x_end, x_start, b)
    elif a < x_start and x_start <= x_end and x_end <= b:
        return integral_of_g(x_start, x_end, x_start, x_end) + integral_of_g(x_start, x_end, x_end, b)
    elif x_start <= a and a < x_end and x_end < b:
        return integral_of_g(x_start, x_end, a, x_end) + integral_of_g(x_start, x_end, x_end, b)
    elif x_start <= a and a <= b and b <= x_end:
        return 1/2.0 * (1/3.0 * (pow(b - x_start, 3) - pow(a - x_start, 3)))
    elif x_start <= x_end and x_end <= a and a < b:
        return 1/2.0 * pow(x_end - x_start, 2) * (b - a) + 1/2.0 * (x_end - x_start) * (pow(b - x_end, 2) - pow(a - x_end, 2))
    else:
        raise NotImplementedError(f"Cannot process these values provided "
                                   f"by x_start: {x_start} and x_end: {x_end} "
                                  f"a: {a} and b: {b}")



class VariationalFramework(object):
    """
    Provides the Variational Framework that will be used to calculate various
    elements of the market.

    The functions will be provided ids ranging from:
        - 1 to number of functions.
    """

    def __init__(self,
                 integral_constraints = [],
                 number_of_functions = 0,
                 start_time: float = 0,
                 end_time: float = 100):

        self.integral_constraints = integral_constraints
        self.number_of_functions = number_of_functions

        if start_time != 0.0:
            raise ValueError("Currently requiring start time to be 0.")


        self.global_min_time = 0
        self.global_max_time = end_time


        self.number_of_constraints = len(integral_constraints)
        self.start_time = start_time
        self.end_time = end_time

        self.function_i_to_constraint_elements = dict()
        self.constraint_elements_for_constraint_i = None
        self.constraint_elements_corresponding_to_function_id = None


        #Validate Setup properly.

        self._validate_incoming_arguments()
        self._initialize_function_i_to_constraint_elements()
        self._validate_variational_framework_setup()

        self.solver_successful = False

    def _validate_variational_framework_setup(self):
        """
        Validate variational framework setup.

        """


        if (self.constraint_elements_for_constraint_i is None
            and self.constraint_elements_corresponding_to_function_id is None):
            raise RuntimeError("Constraint Dictionaries not correctly initialized")


    def _validate_incoming_arguments(self):
        """
        Validate incoming constraints, and arguments.

        The constraints that need to be checked are numerous. Provided below are
        some of the said constraints.

        These constraints include:
            1. Ensuring that all functions are represented in the problem
            2. Checking that minimum and maximum times are respected.

        """

        functions_represented_in_integral_constraints = set()
        for integral_constraint in self.integral_constraints:
            for integral_constraint_element in integral_constraint.integral_constraint_elements:
                function_id_in_integral_constraint = integral_constraint_element.function_id
                functions_represented_in_integral_constraints.add(function_id_in_integral_constraint)



        expected_function_ids = set([i for i in range(1,self.number_of_functions + 1)])
        missing_function_ids = expected_function_ids.symmetric_difference(functions_represented_in_integral_constraints)

        if len(missing_function_ids) > 0:
            raise RuntimeError(f"The User states which functions are expected via the number of constraints. The expected "
                               f" functions are 1...number_of_constraints. The integral constraints specify a set of "
                               f"functions, which may not include every function id. Hence, there could be functions that"
                               f"are not constrained via the integral constraints. In order to avoid these situations, I will "
                               f"throw an error. Every function must be constrained by an integral constraint. An error was thrown"
                               f" here because a function was not constrained properly. The functions that were not constrained are: "
                               f"{missing_function_ids} ")


        #Another constraint that we hoped to enforce was a time constraint.
        #The time constraint looks to ensure that the integral constraints provide constraints
        #between the period of [0,N]. To check this calculate the min time and calculate the max time.
        #After calculating the min time and the max time. Ensure that all integral constraiant elements
        #fall during this period of time.

        min_time = None
        max_time = None
        for integral_constraint in self.integral_constraints:
            for integral_constraint_element in integral_constraint.integral_constraint_elements:

                if integral_constraint_element.start_time > integral_constraint_element.end_time:
                    raise RuntimeError(f"Start Time: {integral_constraint_element.start_time} "
                                       f"End Time: {integral_constraint_element.end_time}. "
                                       f"Start Time cannot come after end time.")

                function_id = integral_constraint_element.function_id
                start_time = integral_constraint_element.start_time
                end_time = integral_constraint_element.end_time

                if min_time is None:
                    min_time = start_time
                else:
                    min_time = min(min_time, start_time)

                if max_time is None:
                    max_time = end_time
                else:
                    max_time = max(max_time, end_time)

        if min_time < self.global_min_time or max_time > self.global_max_time:
            raise RuntimeError(f"The min time of the constraints is {min_time} and the max time of the constraints"
                               f"is {max_time}. The global minimum time allowed is {self.global_min_time} and the "
                               f"global maximum time is {self.global_max_time}. Some constraint does not respect "
                               f"the global minimum and the global maximum.")




    def _initialize_function_i_to_constraint_elements(self):
        """
        Initialize the function i to constraint elements.

        Function needs to implement and initialize two different variables.

        The variables are provided below:
            1. constraint_elements_for_constraint_i
            2. constraint_elements_corresponding_to_function_id

        :return:
        """

        constraint_elements_for_constraint_i = dict()
        for integral_constraint in self.integral_constraints:
            constraint_elements_for_constraint_i[integral_constraint.constraint_id] = integral_constraint.integral_constraint_elements
        self.constraint_elements_for_constraint_i =constraint_elements_for_constraint_i


        constraint_elements_corresponding_to_function_id_dict = dict()
        for integral_constraint in self.integral_constraints:
            for integral_constraint_element in integral_constraint.integral_constraint_elements:
                function_id = integral_constraint_element.function_id
                constraint_elements_corresponding_to_function_id = constraint_elements_corresponding_to_function_id_dict.get(function_id, [])
                constraint_elements_corresponding_to_function_id.append(integral_constraint_element)
                constraint_elements_corresponding_to_function_id_dict[function_id] = constraint_elements_corresponding_to_function_id

        self.constraint_elements_corresponding_to_function_id = constraint_elements_corresponding_to_function_id_dict

        self.constraint_id_to_constraint = dict()
        for integral_constraint in self.integral_constraints:
            self.constraint_id_to_constraint[integral_constraint.constraint_id] = integral_constraint

    def is_solve_successful(self):
        return self.solver_successful

    def get_constraint_elements_for_constraint_i(self) -> dict:
        return self.constraint_elements_for_constraint_i

    def get_constraint_elements_corresponding_to_function_id(self):
        return self.constraint_elements_corresponding_to_function_id


    def form_coefficient_format(self):
        """
        Forms a relevant coefficient format that can be used in the future to
        speak to where various coefficients will get placed etc.

        :return:
        """

        pass

    def get_lambdas_name(self, lambda_id):
        return f"lambdas_{lambda_id}"


    def get_function_evaluated_at_time(self,
                                       time: float,
                                       function_id: int) -> float:
        """
        Gets the function provided by function_id, and then evaluates at
        a particular time.

        """

        if function_id in self.function_id_to_function_lambdas:
            function = self.function_id_to_function_lambdas.get(function_id)
            value = function(time)
            return value
        else:
            raise ValueError(f"Function ID {function_id} not found in function_id_to_function_lambdas.")


    def get_function_integrated(self,
                                start_time: float,
                                end_time: float,
                                function_id: int) -> float:
        """
        Gets the integral of the function provided by function_id from
        start time to end time.

        """

        if function_id in self.function_id_to_function_lambdas:
            function = self.function_id_to_function_lambdas.get(function_id)
            integral_val, _ = integrate.quad(function, start_time, end_time)
            return integral_val
        else:
            raise ValueError(f"Function ID {function_id} not found in function_id_to_function_lambdas.")


    def get_function_ids(self) -> list[int]:
        """
        A set of function ids exist. These function ids lay out what has been solved for.

        It can be used by user.


        :return:
        """

        function_ids = list(self.function_id_to_function_lambdas.keys())
        return function_ids


    def form_matricies(self):
        """
        Solves for a set of continuous functions. The continuous functions will provide
        information on what values were taken on at any given point of time.

        Below, we are presented with a matrix A, x, b.

        Matrix Coefficients are provided below by:
        -----------------------------------------

        lambda_1
        lambda_2
        ...
        lambda_C
        A_1
        B_1
        ...
        A_n
        B_n

        We have the matrix A provided by:
        -----------------------------------------

        Matrix A                                    x                           b
        --------                                   ---                         ---


        lam_1       lam_C A_1, B_1   A_N, B_N
        [                                   ]          []                           []
        [                                   ]          []                           []
        [                                   ]          []                           []
        [                                   ]          []                           []
        [                                   ]          []                           []
        [                                   ]          []                           []

        TODO: Look to add accessors for various coefficients.
        TODO:
        """

        N = self.number_of_constraints + 2 * self.number_of_functions

        A = np.zeros((N, N))
        y = np.zeros((N, 1))

        is_left_endpoint = True
        function_id_for_endpoint = 1
        for i in range(N):
            if i < self.number_of_constraints:

                constraint_id = i + 1
                constraint_elements_corresponding_to_constraint_i = (self.get_constraint_elements_for_constraint_i().
                                                                     get(constraint_id))
                constraint = self.constraint_id_to_constraint.get(constraint_id)
                constraint_value = constraint.integral_value

                y[i] = constraint_value

                logging.info(f"Constraint ID is provided by: {constraint_id}")
                logging.info(f"Constraint Value is provided by: {constraint_value}")

                #Enforcing a particular integral constraint. Each constraint is made up of a set of
                #constraint elements that we will look to integrate over here.
                for constraint_element_corresponding_to_constraint_i in constraint_elements_corresponding_to_constraint_i:



                    governing_constraint_start_time = constraint_element_corresponding_to_constraint_i.start_time
                    governing_constraint_end_time = constraint_element_corresponding_to_constraint_i.end_time

                    #Below, the function_id is the function that will be integrated over.

                    function_id = constraint_element_corresponding_to_constraint_i.function_id
                    constraint_element_constraint_id = constraint_element_corresponding_to_constraint_i.constraint_id
                    constraint_elements_corresponding_to_function_id = (self.get_constraint_elements_corresponding_to_function_id()
                                                                        .get(function_id))

                    logging.info(f"Governing Constraint Start Time: {governing_constraint_start_time}")
                    logging.info(f"Governing Constraint End Time: {governing_constraint_end_time}")
                    logging.info(f"Governing Constraint Function ID: {function_id}")
                    logging.info(f"Constraint ID of Constraint Element is provided by: {constraint_element_constraint_id}")

                    # The integration of function_id between governing constraint start time to
                    # governing end time includes integrating over a certain set of constraint elements.
                    # All the constraint elements corresponding to function id must have function id
                    # as a core function. This may need to be refactored. Visualize the math in one heads can help sig
                    for constraint_element_corresponding_to_function_id in constraint_elements_corresponding_to_function_id:

                        j = constraint_element_corresponding_to_function_id.constraint_id - 1

                        logging.info(f"J is provided by {j}")

                        constraint_element_start_time = constraint_element_corresponding_to_function_id.start_time
                        constraint_element_end_time = constraint_element_corresponding_to_function_id.end_time
                        constraint_element_function_id = constraint_element_corresponding_to_function_id.function_id



                        logging.info(f"Processing a governing constraint element representing the integral of function "
                                     f"with function id {function_id}, from the start time provided by: "
                                     f"{governing_constraint_start_time} and end time {governing_constraint_end_time}. "
                                     f"As part of this calculation, there is also a constraint element corresponding "
                                     f"to the calculated function id, the constraint element start time is provided by: "
                                     f" {constraint_element_start_time} and end time {constraint_element_end_time}, with "
                                     f" a constraint element function id provided by {constraint_element_function_id}.")



                        assert(constraint_element_function_id == function_id)

                        lambda_coefficient = integral_of_g(constraint_element_start_time,
                                                           constraint_element_end_time,
                                                           governing_constraint_start_time,
                                                           governing_constraint_end_time
                                                           )



                        # Integrate and add the lambda coefficients
                        logging.info(f"Integral Of G with the following inputs are. "
                                     f"Constraint Start Time: {constraint_element_start_time}, "
                                     f"Constraint End Time: {constraint_element_end_time}, "
                                     f"Governing Constraint Start Time: {governing_constraint_start_time}, "
                                     f"Governing Constraint End Time {governing_constraint_end_time} and the "
                                     f"lambda coefficient is provided by: {lambda_coefficient}")

                        A[i, j] += 1/2.0 * lambda_coefficient


                    #Integrate A_I coefficient correctly.
                    #int_(gov_start_time)^(gov_end_time) A_I * x = 1/2 * A_I * (gov_end_time^2 - gov_start_time^2)
                    a_i_coefficient = (1/2.0 * pow(governing_constraint_end_time, 2) -
                                      1/2.0 * pow(governing_constraint_start_time, 2))


                    a_i_index = self.number_of_constraints + 2 * (function_id - 1)
                    A[i, a_i_index] += a_i_coefficient


                    # Integrate B_I coefficient correctly.
                    # int_(gov_start_time)^(gov_end_time) B_I = B_I * (gov_end_time - gov_start_time)
                    b_i_coefficient = (governing_constraint_end_time - governing_constraint_start_time)
                    b_i_index = self.number_of_constraints + 2 * (function_id - 1) + 1
                    A[i, b_i_index] += b_i_coefficient

            elif i >= self.number_of_constraints:
                #Looks to constrain either A_I or B_I correctly.
                #There are left and right endpoints. Each endpoint needs to be constrained correctly.

                if is_left_endpoint:
                    #Looks to constraint slope of left endpoint of interval to 0.
                    A[i, i] = 1
                    is_left_endpoint = False
                else:
                    constraint_elements_corresponding_to_function_id = (self.get_constraint_elements_corresponding_to_function_id()
                                                                        .get(function_id_for_endpoint))


                    for constraint_element_corresponding_to_function_id in constraint_elements_corresponding_to_function_id:

                        function_id = constraint_element_corresponding_to_function_id.function_id
                        start_time = constraint_element_corresponding_to_function_id.start_time
                        end_time = constraint_element_corresponding_to_function_id.end_time
                        constraint_i = constraint_element_corresponding_to_function_id.constraint_id
                        evaluation_time = self.end_time
                        A[i, constraint_i - 1] += 1/2.0 * h(evaluation_time,
                                                             start_time,
                                                             end_time)



                    A[i, i-1] += 1
                    is_left_endpoint = True
                    function_id_for_endpoint += 1
            else:
                raise NotImplementedError("Cannot calculate constraint for i greater "
                                          "than self.integral_constraints")

        return A, y



    def debug_A_matrix(self, A):

        # We can look at the rank of various components of the matrix.
        begin_row = 1
        end_row = 6
        sub_matrix = A[begin_row:end_row, :]
        rank_A = np.linalg.matrix_rank(sub_matrix)


    def _check_visually(self):
        """
        Checks visually with plots the functions that were previously
        solved for.

        """

        fig, axes = plt.subplots(self.number_of_functions, 1, figsize=(10, 5))

        try:
            for i in range(self.number_of_functions):
                function_i = self.function_id_to_function_lambdas.get(i + 1)
                x = np.linspace(self.global_min_time, self.global_max_time, num=1000)
                f_x = list(map(lambda t: function_i(t), x))
                axes[i].plot(x, f_x)
                axes[i].set_title(f'Function I: {i}')
                axes[i].set_xlabel('Time')
                axes[i].set_ylabel('Function Value')
                plt.savefig("plot_functions.png")
        except:
            pass

    def solve(self,
              check_integral_result=True,
              check_visually=True):
        """
        Solves the interpolation problem for a set of integral
        constraints.

        Checks the result for the interpolation techniques if
        check_result is True, which is the default behaviour.

        It may be good to set check_result to False, so that the code
        runs faster.

        """


        A, y = self.form_matricies()
        m, n = A.shape
        logging.info(f"A Shape. Number Of Rows: {m}, Number of Columns: {n}")

        det = np.linalg.det(A)
        if np.isclose(det, 0) or not math.isfinite(det):
            logging.debug(f"Matrix is singular. Cannot use a direct solve. The matrix is provided "
                          f"by {A}")

            x, _, rank, residuals = np.linalg.lstsq(A, y)
            self.debug_A_matrix(A)

            residuals = y - A @ x

            residual_norm = np.linalg.norm(residuals)
            y_norm = np.linalg.norm(y)
            if not np.isclose(residual_norm, 0.0, atol=1):
                raise RuntimeError(f"Matrix is singular, residuals norm is non-zero. Solve has failed. The residual"
                                   f"norm is provided by {residual_norm} and the y norm is provided by: {y_norm}")

        else:
            try:

                # Run the correct linear solve. Check the residual to see if the residual is close enough to zero.
                # This means that the solution x should allow us to solve the relevant integral
                # constraints.

                x = np.linalg.solve(A, y)
                residual = A @ x - y
                norm_of_residual = np.linalg.norm(residual)
                norm_of_y = np.linalg.norm(y)
                if np.isclose(norm_of_residual, 0.0, rtol=0.01, atol=0.01 * norm_of_y):
                    pass
                else:
                    raise RuntimeError("The matrix is non-singular. The matrix solve completed. The residual "
                                       "norm may not be close enough to zero. ")


            except Exception as e:
                raise NotImplementedError(f"Cannot solve for x with A and b. The error is provided by {str(e)}")


        lambdas, function_id_to_coefficients = self.extract_parameters_from_x(x)

        # We can look at what these matricies look like. Looking at these matricies
        # will be critical.

        logging.info(f"Matrix A is provided by: \n {A}")
        logging.info(f"Matrix Y is provided by: \n {y}")
        logging.info(f"x: \n {x}")
        logging.info(f"Lambdas are provided by: \n {lambdas}")
        logging.info(f"Function ID To Coefficients: \n {function_id_to_coefficients}")


        self.lambdas = lambdas
        self.function_id_to_coefficients = function_id_to_coefficients
        self.function_id_to_function_lambdas = self.calculate_functions()

        if check_integral_result:
            integral_constraints_satisfied = self.check_integral_constraints()
        else:
            integral_constraints_satisfied = None

        if check_visually:
            self._check_visually()

        self.solver_successful = True

        return self.function_id_to_function_lambdas, integral_constraints_satisfied


    def extract_parameters_from_x(self, x) -> dict:
        """
        Extracts parameters from x to become lambdas, A_i, B_i that will
        be used to form the function that will be used.

        The parameters that need to be extracted are the:
            1. lambdas
            2. A_i
            3. B_i

        All of these parameters need to be correctly extracted and then used
        in the correct manner.

        """

        n = x.size
        lambdas = dict()
        function_id_to_coefficients = dict()
        is_a_coefficient = True
        function_id = 1
        for i in range(n):
            if i < self.number_of_constraints:
                constraint_id = i + 1
                lambdas[self.get_lambdas_name(constraint_id)] = x[i]
            else:
                if is_a_coefficient:
                    coefficients = function_id_to_coefficients.get(function_id, dict())
                    coefficients[f"A_{function_id}"] = x[i]
                    function_id_to_coefficients[function_id] = coefficients
                    is_a_coefficient = False
                else:
                    coefficients = function_id_to_coefficients.get(function_id, dict())
                    coefficients[f"B_{function_id}"] = x[i]
                    function_id_to_coefficients[function_id] = coefficients
                    is_a_coefficient = True
                    function_id += 1

        return lambdas, function_id_to_coefficients

    def function_lambda(self,
                        function_id,
                        lambdas,
                        function_id_to_coefficients,
                        t) -> float:

        """
        Takes in parameters and looks to return the relevant results.

        The parameters will inform us on what will be done in the calculation.

        """

        constraint_elements_corresponding_to_function_id = self.constraint_elements_corresponding_to_function_id.get(function_id)

        if constraint_elements_corresponding_to_function_id is None:
            raise RuntimeError(f"Cannot the value for function id {function_id}")

        val = 0
        for constraint_element_corresponding_to_function_id in constraint_elements_corresponding_to_function_id:
            start_time = constraint_element_corresponding_to_function_id.start_time
            end_time = constraint_element_corresponding_to_function_id.end_time
            constraint_id = constraint_element_corresponding_to_function_id.constraint_id
            lambda_i = lambdas.get(self.get_lambdas_name(constraint_id))
            if lambda_i is None:
                raise RuntimeError(f"Cannot find lambda for constraint id {constraint_id}")
            val += lambda_i * g(t, start_time, end_time)


        coefficients_for_function_id = function_id_to_coefficients.get(function_id)
        a_coefficient = coefficients_for_function_id[f"A_{function_id}"]
        b_coefficient = coefficients_for_function_id[f"B_{function_id}"]
        function_val = 1/2.0 * val + a_coefficient * t + b_coefficient

        return function_val


    def calculate_functions(self):
        """
        Looks to calculate the relevant functions on the basis of the determined
        coefficients.


        :return:
        """

        function_id_to_function_lambdas = dict()
        for function_id in range(1, self.number_of_functions + 1):

            function_lambda_with_arguments = partial(self.function_lambda,
                                                     function_id,
                                                     self.lambdas,
                                                     self.function_id_to_coefficients)

            function_id_to_function_lambdas[function_id] = function_lambda_with_arguments


        return function_id_to_function_lambdas

    def check_integral_constraints(self):
        """
        The function looks to check the integral constraints that were provided
        when looking to solve the problem.

        :return:
        """


        if self.function_id_to_function_lambdas is None:
            raise RuntimeError("function_id_to_function_lambdas is None. Need to run the computation first "
                               "to develop calculated function_id_to_function_lambdas ")

        integral_constraint_id_to_satisfaction = dict()
        for integral_constraint in self.integral_constraints:
            actual_integral_value = integral_constraint.integral_value
            integral_constraint_elements = integral_constraint.integral_constraint_elements
            integral_constraint_id = integral_constraint.constraint_id
            calculated_integral_value = 0
            for integral_constraint_element in integral_constraint_elements:
                function_id = integral_constraint_element.function_id
                start_time = integral_constraint_element.start_time
                end_time = integral_constraint_element.end_time

                function_lambda = self.function_id_to_function_lambdas.get(function_id)
                if function_lambda is None:
                    raise RuntimeError(f"Function lambda can not be found for the function id {function_id}")
                integral_val, _ = integrate.quad(function_lambda, start_time, end_time)
                calculated_integral_value += integral_val

            logging.info(f"Calculated Integral Value is: {calculated_integral_value}, and actual integral "
                         f"value is: {actual_integral_value}")
            integral_constraint_satisfied = np.isclose(calculated_integral_value, actual_integral_value, rtol=0.01)
            integral_constraint_id_to_satisfaction[integral_constraint_id] = integral_constraint_satisfied

        constraints_satisfied = all([constraint_satisfied for constraint_satisfied
                                     in integral_constraint_id_to_satisfaction.values()])

        return constraints_satisfied


    def get_start_time(self):
        return self.start_time

    def get_end_time(self):
        return self.end_time

    def get_number_of_functions(self):
        return self.number_of_functions

    def get_number_of_integral_constraints(self):
        return self.integral_constraints


    def get_function(self):
        return self.get_function()



class IntegralConstraintElement():
    """
    Integral Constraint Element that will be used in building the
    IntegralConstraint. The Integral Constraint Element will be used
    to build up the IntegralConstraint.


    """

    def __init__(self,
                 function_id: int,
                 start_time: float,
                 end_time: float,
                 constraint_id: int):
        """
        Provides information on a IntegralConstraint Element that can be used in
        the IntegralConstraint.

        """

        self.function_id = function_id
        self.start_time = start_time
        self.end_time = end_time
        self.constraint_id = constraint_id


class IntegralConstraint():
    """
    Integral Constraints are critical to calculating the continuous functions.

    The integral constraint is made up of the IntegralConstraintElement.

    """

    def __init__(self, integral_constraint_elements,
                       integral_value,
                       integral_constraint_id):

        self.integral_constraint_elements = integral_constraint_elements
        self.integral_value = integral_value
        self.constraint_id = integral_constraint_id

    def get_number_of_constraint_elements(self):
        return len(self.integral_constraint_elements)



def create_and_solve_problem_0():
    """
    Creates a problem that will be used to test the Variational Framework
    that was developed previously.

    """


    #Develop constraint 1

    integral_constraint_value_1 = 10
    integral_constraint_id = 1

    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)



    integral_constraint_elements = [integral_constraint_element_1]

    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)



    integral_constraints = [integral_constraint_1]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     1,
                                     start_time,
                                     end_time)



    result, solve_successful = framework.solve()

    return solve_successful


def create_and_solve_problem_1():
    """
    Creates a problem that will be used to test the Variational Framework
    that was developed previously.

    """


    #Develop constraint 1

    integral_constraint_value_1 = 10
    integral_constraint_id = 1

    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)

    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              2,
                                                              3,
                                                              integral_constraint_id)


    integral_constraint_elements = [integral_constraint_element_1,
                                    integral_constraint_element_2]

    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)



    integral_constraints = [integral_constraint_1]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     2,
                                     start_time,
                                     end_time)



    result, solve_successful = framework.solve()

    return solve_successful


def create_and_solve_problem_2():
    """
    Creates a problem that will be used to test the Variational Framework
    that was developed previously.

    """


    #Develop constraint 2

    integral_constraint_value_1 = 10
    integral_constraint_id = 1
    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)



    integral_constraint_id = 2
    integral_constraint_value_2 = 5
    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              2,
                                                              3,
                                                              integral_constraint_id)


    integral_constraint_elements = [integral_constraint_element_1]
    integral_constraint_id = 1
    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)



    integral_constraint_elements = [integral_constraint_element_2]
    integral_constraint_id = 2
    integral_constraint_2 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_2,
                                               integral_constraint_id)



    integral_constraints = [integral_constraint_1,
                            integral_constraint_2]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     2,
                                     start_time,
                                     end_time)



    result, solve_successful = framework.solve()

    return solve_successful


def create_and_solve_problem_3():
    """
    Creates a problem that will be used to test the Variational Framework
    that was developed previously.


    These problems are:
    --------------------------------------------------------------------------------

        1. There are 2 integral constraint elements.
        2. One integral constraint elements correspond to integral constraint 1.
        3. One integral constraint elements correspond to integral constraint 2.
        4. There are two functions, function id 1 and function id 2.
        5. One integral constraint elements correspond to function id 1.
        6. One integral constraint elements correspond to function id 2.

    From above, we can see that the issues that arise come from the fact that we cannot
    look at multiple constraint elements.

    """


    #Develop integral constraint 1

    integral_constraint_value_1 = 10
    integral_constraint_id = 1
    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)






    integral_constraint_elements = [integral_constraint_element_1]
    integral_constraint_id = 1
    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)

    #Develop integral constraint 2.

    integral_constraint_id = 2
    integral_constraint_value_2 = 5
    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              2,
                                                              3,
                                                              integral_constraint_id)


    integral_constraint_elements = [integral_constraint_element_2]
    integral_constraint_id = 2
    integral_constraint_2 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_2,
                                               integral_constraint_id)



    integral_constraints = [integral_constraint_1,
                            integral_constraint_2]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     2,
                                     start_time,
                                     end_time)



    result, solve_successful = framework.solve()

    return solve_successful

def create_and_solve_problem_4():
    """
    Creates a problem that will be used to test the Variational Framework
    that was developed previously.


    Notes:
    ----------------------------------------------------------------------

    1. The problem is that when we have multiple constraints and then for each
        constraint we have multiple constraint elements.

    2. The above case is the one that we cannot handle properly.

        - How are we going to find the correct values.


    Below, we look to map the problem that is solved below.
        1. I look to solve a certain problem below.

    What is the problem that we solve below. We have a number of different things
    that get solved below.

    These problems are:
    --------------------------------------------------------------------------------

        1. There are 4 integral constraint elements.
        2. Two integral constraint elements correspond to integral constraint 1.
        3. Two integral constraint elements correspond to integral constraint 2.
        4. There are two functions, function id 1 and function id 2.
        5. Two integral constraint elements correspond to function id 1.
        6. Two integral constraint elements correspond to function id 2.

    """


    #Develop constraint 1

    integral_constraint_value_1 = 10
    integral_constraint_id = 1

    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)

    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              2,
                                                              3,
                                                              integral_constraint_id)


    integral_constraint_elements = [integral_constraint_element_1,
                                    integral_constraint_element_2]

    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)

    # Develop constraint 2.

    #Maybe we need to change the vector we are looking
    #to approximate

    integral_constraint_value_2 = -5
    integral_constraint_id = 2

    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              4,
                                                              5,
                                                              integral_constraint_id)

    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              6,
                                                              8,
                                                              integral_constraint_id)


    integral_constraint_elements = [integral_constraint_element_1, integral_constraint_element_2]

    integral_constraint_2 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_2,
                                               integral_constraint_id)


    #Place integral constraints in the VariationalFramework.

    integral_constraints = [integral_constraint_1,
                            integral_constraint_2]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     2,
                                     start_time,
                                     end_time)



    result, solve_successful = framework.solve()

    return solve_successful


def create_and_solve_problem_5():
    """
    Creates a problem that will then be solved with the Variational Framework.

    Goal here is to look at the an integral constraint which is offset in time.


    :return:
    """

    integral_constraint_value_1 = 10
    integral_constraint_id = 1

    integral_constraint_element_1 = IntegralConstraintElement(1,
                                                              0,
                                                              3,
                                                              integral_constraint_id)

    integral_constraint_element_2 = IntegralConstraintElement(2,
                                                              0,
                                                              3,
                                                              integral_constraint_id)

    integral_constraint_elements = [integral_constraint_element_1,
                                    integral_constraint_element_2]

    integral_constraint_1 = IntegralConstraint(integral_constraint_elements,
                                               integral_constraint_value_1,
                                               integral_constraint_id)

    integral_constraints = [integral_constraint_1]


    start_time = 0
    end_time = 10
    framework = VariationalFramework(integral_constraints,
                                     2,
                                     start_time,
                                     end_time)

    result, solve_successful = framework.solve()




if __name__ == '__main__':

    logging.info("Running main function")

    create_and_solve_problem_5()
















