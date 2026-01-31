import yaml
import argparse
import sys

sys.path.insert(0, "src")
from flax_training_engine import (
    MAJOR_VERSION,
    MINOR_VERSION as TE_MINOR_VERSION,
    PATCH_VERSION,
    TWEAK_VERSION,
)

parser = argparse.ArgumentParser(description="Update experiment version in YAML file")
parser.add_argument("--commit_id", type=str, help="The git commit id", required=True)
parser.add_argument(
    "--commit_msg", type=str, help="The git commit message", required=True
)
args = parser.parse_args()

# 1) Load the latest data engine version from data-experiments.yaml
with open("data-experiments.yaml", "r") as file:
    data_experiments = yaml.safe_load(file)

if data_experiments is None:
    data_experiments = {}

# Find latest data version
latest_data_version = 0
for key in data_experiments.keys():
    try:
        version_num = int(key.split("_")[-1])
        if version_num > latest_data_version:
            latest_data_version = version_num
    except (ValueError, IndexError):
        continue

# Get the data-version string from the latest data experiment
if latest_data_version > 0 and f"data_{latest_data_version}" in data_experiments:
    data_version_str = data_experiments[f"data_{latest_data_version}"]["version"]
else:
    data_version_str = "0.0.0.0"

# Read the experiments YAML file
with open("experiments.yaml", "r") as file:
    data = yaml.safe_load(file)

# Handle empty file
if data is None:
    data = {}

# Find latest experiment version
latest_version = 0
for key in data.keys():
    try:
        version_num = int(key.split("_")[-1])
        if version_num > latest_version:
            latest_version = version_num
    except (ValueError, IndexError):
        continue

# 2) Generate the new experiment version by incrementing the minor version of the previous one
if latest_version > 0 and f"experiment_{latest_version}" in data:
    version = data[f"experiment_{latest_version}"]["version"]
    version_keys = version.split(".")
    new_version_str = f"{version_keys[0]}.{int(version_keys[1]) + 1}.{version_keys[2]}.{version_keys[3]}"
else:
    new_version_str = "0.1.0.0"

# 3) Use the description as the commit message (argument)
# 4) Use the git hash as the commit id (argument)
new_version = latest_version + 1
training_engine_version = (
    f"{MAJOR_VERSION}.{TE_MINOR_VERSION}.{PATCH_VERSION}.{TWEAK_VERSION}"
)
data[f"experiment_{new_version}"] = {
    "version": new_version_str,
    "data-version": data_version_str,
    "training-engine-version": training_engine_version,
    "description": args.commit_msg,
    "git-hash": args.commit_id,
}

# Write updated data back to file
with open("experiments.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

with open("src/config.py", "r+") as config_file:
    minor_version = new_version_str.split(".")[1]
    lines = config_file.readlines()
    config_file.seek(0)
    config_file.truncate()

    found = False
    for line in lines:
        if line.startswith("MINOR_VERSION"):
            config_file.write(f"MINOR_VERSION = {minor_version}\n")
            found = True
        else:
            config_file.write(line)

    if not found:
        config_file.write(f"MINOR_VERSION = {minor_version}\n")
