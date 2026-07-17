"""



"""
from datetime import datetime
from calendar import monthrange

def get_last_date_of_month(date: str) -> str:

    date_dt = datetime.strptime(date, "%Y-%m-%d")
    day = date_dt.day
    month = date_dt.month
    year = date_dt.year
    end_day = monthrange(year, month)[1]
    return datetime(year, month, end_day).strftime("%Y-%m-%d")


def get_number_days_in_month(year, month):
    return monthrange(year, month)[1]