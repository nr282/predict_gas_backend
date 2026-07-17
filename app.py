# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#Test

import signal
import sys
from types import FrameType

from flask import Flask, request, jsonify

from utils.logging import logger
from variational_framework.daily_cash_flow import calculate_daily_cash_flow
import pandas as pd
app = Flask(__name__)


def _validate(dates, cashflows):
    return True

@app.route("/")
def calculate_cash_flows() -> str:
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # Use request.args.get() to safely pull values (returns None if missing)
    values = request.args.get('values')
    start_quarter = request.args.get('start_quarter')
    wacc_quarterly = request.args.get('wacc')
    wacc_quarterly = float(wacc_quarterly)
    wacc_daily = (1 + wacc_quarterly) ** (1/91.25) - 1

    print("Values are provided by")
    print(values)

    print("Begin Quarter: ")
    print(start_quarter)

    values = [float(val) for val in values.split(",")]

    n = len(values)
    quarters = pd.period_range(start=start_quarter, periods=n, freq="Q-DEC")
    dates = quarters.to_timestamp()

    data = pd.DataFrame.from_dict({"Date": dates, "Quarters": quarters, "Value": values})
    data["quarter_number"] = range(len(quarters))
    data["quarter_discount"] = data["quarter_number"].apply(lambda qn: (1-wacc_quarterly)**qn)
    data["quarter_discounted"] = data.apply(lambda row: row["quarter_discount"] * row["Value"], axis=1)
    quarterly_dcf = data["quarter_discounted"].sum()

    result = calculate_daily_cash_flow(data)
    result["wacc_quarterly"] = wacc_quarterly
    result["wacc_daily"] = wacc_daily
    result["day_number"] = range(len(result))
    #Interet rate:
    #(1 - ((1 + 0.1) ** (1/91.25) - 1))^90

    result["discount"] = result["day_number"].apply(lambda dn: (1-wacc_daily)**dn)
    result["discounted_cash"] = result.apply(lambda row: row["discount"] * row["Value"], axis=1)
    dcf = result["discounted_cash"].sum()
    result["advanced_dfc_value"] = dcf
    result["quarterly_dcf"] = quarterly_dcf


    res = result.to_dict()
    return res


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
