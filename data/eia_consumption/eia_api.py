"""
EIA API is of high importance



"""

import requests
import logging

def read_eia_path(eia_api_path):
    """
    Reads EIA Data from eia_api_path.

    :return:
    """

    eia_result = None
    try:
        eia_result = requests.get(eia_api_path)
    except Exception as e:
        logging.info(f"Error found for the exception provided by {str(e)}")
        raise RuntimeError("Cannot get result ")


    request_call_successful = False
    if not eia_result.ok:
        if eia_result.status_code == 403:
            raise RuntimeError(f"Error provided by: {eia_result.reason}. This likely due to "
                               f"not using the API Key. An individual can acquire API key from the EIA website.")
        elif eia_result.status_code == 200:
            request_call_successful = True
        else:
            raise NotImplementedError(f"Cannot process a specific status code. The status code is provided by "
                                      f"{eia_result.status_code}")
    else:
        request_call_successful = True
        assert(eia_result.status_code == 200)

    return request_call_successful, eia_result