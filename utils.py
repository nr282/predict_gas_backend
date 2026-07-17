import os


def get_base_path():

    potential_paths = ["C:\\Users\\nr282\\PycharmProjects\\PythonProject4",
                       "C:\\Users\\nr282\\PycharmProjects\\natural_gas_consumption",
                       "/home/ec2-user/natural_gas_consumption",
                       "/var/task/"] #/var/task is the lambda deployment.

    actual_path = None
    for pot_path in potential_paths:
        if os.path.exists(pot_path):
            actual_path = pot_path
            break

    if actual_path is None:
        raise ValueError("Could not find base path")

    return actual_path

def write_file_to_directory(suffix_path: str):
    pass