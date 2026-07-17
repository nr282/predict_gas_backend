"""
A major goal of the model module is to develop a class that represents a module.

From initial investigation, the parameters and the inference code should live in the same
location.
"""


from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
from optimization import grid_search
import numpy as np
from scipy.optimize import dual_annealing
import logging
import time

UPPER_MULTIPLICATIVE_BOUND = 2


class Model(ABC):
    """
    States the parameters and the methods that will be housed in the model.

    A key element of this is that:
        1. parameters in the model. The seven parameters that define the model.
        2. functions to do the inference, such as infer_residential.

    """

    def __init__(self, calibrated_parameters, parameter_list):
        self.calibrated_parameters = calibrated_parameters
        self.parameter_list = parameter_list


    @abstractmethod
    def inference(self,
                  start_datetime: datetime,
                  end_datetime: datetime,
                  params: dict,
                  data: dict) -> Tuple[dict, float]:
        pass

    @abstractmethod
    def get_params_for_model(self) -> dict:
        pass

    def calculate_accuracy(self,
                           estimated_monthly_data,
                           data,
                           state):

        eia_data = data["full_eia_data"]
        merged_df = eia_data.merge(estimated_monthly_data, on="Date")
        merged_df["error"] = (merged_df[state].astype(np.float64) - merged_df["eia_observations"]).abs()
        merged_df["relative_error_non_percent"] = merged_df["error"] / merged_df[state].astype(np.float64).abs()
        logging.info(f"Monthly Estimate Comparison {merged_df}")
        logging.info(f"Monthly Estimate Comparison {merged_df[['relative_error_non_percent', 'Date']]}")
        return float(merged_df["error"].sum() / merged_df[state].astype(np.float64).abs().sum())

    @abstractmethod
    def inference(self,
                start_datetime: str,
                end_datetime: str,
                eia_start_time: str,
                eia_end_time: str,
                params: dict,
                data: dict) -> Tuple[dict, float]:
        """
        Abstracts the inference model.

        This is the core function that looks to infer the relevant parameters.
        """
        pass

    def override_parameters(self):

        base_parameters = self.get_params_for_model()
        param_to_value = dict()
        for param in base_parameters:
            if param in self.calibrated_parameters:
                param_to_value[param] = self.calibrated_parameters[param]
            else:
                param_to_value[param] = base_parameters.get(param)
        return param_to_value


    def convert_x_to_params(self, x):
        params = dict()
        param_names = self.get_params_for_model().keys()
        for i, param_name in enumerate(param_names):
            params[param_name] = x[i]
        return params

    def convert_params_to_x(self, params):

        n = len(params)
        x = np.zeros(n)
        for i, param in enumerate(params):
            x[i] = params[param]
        return x

    def calculate_bounds(self):
        param_names = self.get_params_for_model().keys()
        n = len(param_names)
        bounds = []
        param_to_value = self.override_parameters()
        for param_name in param_names:
            #upper_bound = UPPER_MULTIPLICATIVE_BOUND * param_to_value.get(param_name)
            bounds.append((0.8 * param_to_value.get(param_name), 1.2 * param_to_value.get(param_name)))
        return bounds

    def global_optimize(self,
                        start_datetime: str,
                        end_datetime: str,
                        eia_start_time: str,
                        eia_end_time: str,
                        data: dict,
                        app_params: dict = None):
        """
        Globally optimizes the model parameters.
        """

        def func(x):
            params = self.convert_x_to_params(x)
            estimated_daily_data, estimated_monthly_data, params = self.inference(start_datetime,
                                                                                   end_datetime,
                                                                                   eia_start_time,
                                                                                   eia_end_time,
                                                                                   params,
                                                                                   data,
                                                                                   app_params=app_params)

            if estimated_daily_data is None or estimated_monthly_data is None or params is None:
                return float('inf')

            error = self.calculate_accuracy(estimated_monthly_data, data, data["state"])
            log_handler = app_params["log_handler"]
            file_handler = app_params["file_handler"]
            log_handler.info(f"The relative error is {error} for params {params}")
            time.sleep(1)
            file_handler.flush()
            return error

        bounds = self.calculate_bounds()

        ret = dual_annealing(func, bounds=bounds)
        return self.convert_x_to_params(ret.x), ret.fun


    def run_inference_engine_with_global_optimization(self,
                                                     start_datetime: str,
                                                     end_datetime: str,
                                                     eia_start_time: str,
                                                     eia_end_time: str,
                                                     data: dict,
                                                     app_params: dict = None) -> Tuple[dict, float]:
        """
        The goal of the run inference engine with global optimization is to find the optimal
        parameters for the function associated with self.inference_model which is an abstract
        function provided above.

        """

        params, opt_relative_error = self.global_optimize(start_datetime,
                                                         end_datetime,
                                                         eia_start_time,
                                                         eia_end_time,
                                                         data,
                                                         app_params)
        return params, opt_relative_error

    def run_inference_engine(self,
                             start_datetime: str,
                             end_datetime: str,
                             eia_start_time: str,
                             eia_end_time: str,
                             params: dict,
                             data: dict) -> Tuple[dict, float]:

        #Gather Base Parameters
        base_parameters = self.get_params_for_model()
        param_to_value = self.override_parameters()


        parameter_grid, _ = grid_search.generate_grid(param_to_value)
        optimal_param = None
        optimal_val = None
        relative_error = None
        for param_to_value in parameter_grid():
            estimated_daily_data, estimated_monthly_data, params = self.inference(start_datetime,
                                                                                 end_datetime,
                                                                                 eia_start_time,
                                                                                 eia_end_time,
                                                                                 param_to_value,
                                                                                 data)

            relative_error = self.calculate_accuracy(estimated_monthly_data, data, data["state"])

            logging.info(f"The percent error is {100 * relative_error} for params {params}")
            logging.info(f"The estimated monthly data is: {estimated_monthly_data}")
            logging.info(f"The actual data is: {data['full_eia_data']}")

            if optimal_param is None:
                optimal_param = param_to_value
                optimal_val = relative_error
            else:
                if relative_error < optimal_val:
                    optimal_param = param_to_value
                    optimal_val = relative_error

        return optimal_param, relative_error

