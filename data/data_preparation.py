"""
Gathers all of the relevant data, and loads the data.

"""


def gather_relevant_data(config):
    """
    The function gathers the relevant data that is passed in via the config file.

    """

    datasets = dict()
    for dataset in config["Datasets"]:
        if dataset == "EIA":
            pass
        elif dataset == "Weather":
            pass
        elif dataset == "Wind":
            pass
        elif dataset == "Population":
            pass
        else:
            raise NotImplementedError(f"Cannot process dataset provided by {dataset}")
        datasets[dataset] = None

    return datasets