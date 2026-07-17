import logging
logging.basicConfig(level=logging.INFO)


us_state_to_abbrev = {
        "Alabama": "AL",
        "Alaska": "AK",
        "Arizona": "AZ",
        "Arkansas": "AR",
        "California": "CA",
        "Colorado": "CO",
        "Connecticut": "CT",
        "Delaware": "DE",
        "Florida": "FL",
        "Georgia": "GA",
        "Hawaii": "HI",
        "Idaho": "ID",
        "Illinois": "IL",
        "Indiana": "IN",
        "Iowa": "IA",
        "Kansas": "KS",
        "Kentucky": "KY",
        "Louisiana": "LA",
        "Maine": "ME",
        "Maryland": "MD",
        "Massachusetts": "MA",
        "Michigan": "MI",
        "Minnesota": "MN",
        "Mississippi": "MS",
        "Missouri": "MO",
        "Montana": "MT",
        "Nebraska": "NE",
        "Nevada": "NV",
        "New Hampshire": "NH",
        "New Jersey": "NJ",
        "New Mexico": "NM",
        "New York": "NY",
        "North Carolina": "NC",
        "North Dakota": "ND",
        "Ohio": "OH",
        "Oklahoma": "OK",
        "Oregon": "OR",
        "Pennsylvania": "PA",
        "Rhode Island": "RI",
        "South Carolina": "SC",
        "South Dakota": "SD",
        "Tennessee": "TN",
        "Texas": "TX",
        "Utah": "UT",
        "Vermont": "VT",
        "Virginia": "VA",
        "Washington": "WA",
        "West Virginia": "WV",
        "Wisconsin": "WI",
        "Wyoming": "WY"
    }


us_state_to_abbrev_supported_for_trading = {
            "California": "CA",
            "New York": "NY",
            "Florida": "FL",
            "Texas": "TX",
            "Pennsylvania": "PA",
            "Illinois": "IL"
}

us_state_to_abbrev_supported_by_prescient = {
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY"
}

# invert the dictionary
abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))



def convert_native_name_to_standard_state_name(native_name):
    """
    Converts native name to standard state name.

    :param native_name:
    :return:
    """

    if native_name.startswith("USA"):
        hyphen_loc = native_name.find("-")
        if hyphen_loc != -1:
            native_name_abbreviation = native_name[hyphen_loc + 1:]
            if native_name_abbreviation in abbrev_to_us_state:
                us_state_name = abbrev_to_us_state[native_name_abbreviation]
                return us_state_name
            else:
                logging.critical(f"State Name cannot be found for Native Name Abbreviation. The abbreviation is: {native_name_abbreviation}")
    elif native_name == "U.S.":
        return "United States"
    else:
        if native_name.title() in us_state_to_abbrev:
            return native_name.title()
        else:
            raise ValueError("State Name cannot be found for Native Name Abbreviation. ")

def get_fifty_us_states_and_dc():

    states_and_territories_set = set(list(us_state_to_abbrev.keys()))

    territories = ["American Samoa",
                   "Northern Mariana Islands",
                   "Puerto Rico",
                   "United States Minor Outlying Islands",
                   "Virgin Islands, U.S.",
                   "Guam"
                   ]

    territories_set = set(territories)

    states_set = states_and_territories_set - territories_set
    return states_set

def get_united_states_name():
    return "United States"



