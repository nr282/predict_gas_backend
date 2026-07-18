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
from baseline.baseline import calculate_eia_daily_values, ComponentType
from flask import Flask, request, jsonify

from utils.logging import logger

app = Flask(__name__)


def _validate(dates, cashflows):
    return True

@app.route("/")
def calculate_daily_natural_gas_consumption_values() -> str:
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # Use request.args.get() to safely pull values (returns None if missing)
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    current_date = request.args.get('current_date')
    state = request.args.get('state')

    daily_values = calculate_eia_daily_values(start_date,
                                              end_date,
                                              "2009-01-01",
                                              "2025-09-30",
                                              "2000-01-01",
                                              "2008-12-31",
                                              current_date,
                                              ComponentType.RESIDENTIAL,
                                              state)

    res = daily_values.to_dict()
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
