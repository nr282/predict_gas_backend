"""
Mathematical Models for Natural Gas Forecasting.

The major paper is provided here:
    1. https://epublications.marquette.edu/cgi/viewcontent.cgi?article=1291&context=electric_fac
    2. https://www.ice.com/white-paper/natural-gas-market-storage-dynamics-and-alpha-generation
    3. https://pmc.ncbi.nlm.nih.gov/articles/PMC12116050/pdf/sensors-25-03079.pdf

From the above papers, I list out the core elements to consider:
    1. Five Factor Model
    2. HDD
    3. CDD
    4. Day of Week
    5. Wind Adjusted HDD
    6. Bill Shock
    7. Occupancy Rates
    8. Industrial Production
    9. Economic Factors
    10. Industrial Factors
    11. Weekday
    12. Weekend

In the above paper, we can look to calculate the CDD/HDD.

The formulas are presented below:
    1. HDD_k = max(0,T_ref - T_k)
        - T_ref = 65 degrees Farenheit or 55 degrees Farenheit.
        - 18 degrees Celcius

"""

from enum import Enum

class TemperatureType(Enum):
    CELCIUS = 1
    FARENHEIT = 2


def celcius_to_farenheit(celcius):
    return 9/5 * celcius + 32

def farenheit_to_celcius(farenheit):
    return (farenheit - 32) * 5/9

def farenheit_to_hdd(farenheit):
    return max(65 - farenheit, 0)

def farenheit_to_cdd(farenheit):
    return max(farenheit - 65, 0)

def calculate_hdd(temp, type):
    if type == TemperatureType.CELCIUS:
        f = celcius_to_farenheit(temp)
        hdd = farenheit_to_hdd(f)
        return hdd
    else:
        hdd = farenheit_to_hdd(temp)
        return hdd

def calculate_cdd(temp, type):
    if type == TemperatureType.CELCIUS:
        f = celcius_to_farenheit(temp)
        cdd = farenheit_to_hdd(f)
        return cdd
    else:
        cdd = farenheit_to_hdd(temp)
        return cdd

