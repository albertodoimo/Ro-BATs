import os
import yaml

os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Set the working directory to the script's location
# Specify the path to your YAML file
yaml_file = "calibration_matrix.yaml"

try:
    with open(yaml_file, 'r') as file:
        # Load the YAML content into a Python variable (usually a dict or list)
        data = yaml.safe_load(file)
        print(data['camera_matrix'])
        print(data['dist_coeff'])
    print("YAML data loaded successfully:")
    print(data)
except FileNotFoundError:
    print(f"Error: The file '{yaml_file}' does not exist.")
except yaml.YAMLError as exc:
    print("Error parsing YAML file:", exc)
