import yaml
import argparse
import sys

sys.path.insert(0, "src")
from src.data_pipeline.version import (
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_VERSION,
    TWEAK_VERSION,
)

parser = argparse.ArgumentParser(description="Update data version in YAML file")
parser.add_argument("--commit_id", type=str, help="The git commit id", required=True)
parser.add_argument(
    "--commit_msg", type=str, help="The git commit message", required=True
)
args = parser.parse_args()

# Increment the minor version from version.py
new_minor_version = MINOR_VERSION + 1
new_version_str = f"{MAJOR_VERSION}.{new_minor_version}.{PATCH_VERSION}.{TWEAK_VERSION}"

# Read the YAML file
with open("data-experiments.yaml", "r") as file:
    data = yaml.safe_load(file)

# Handle empty file
if data is None:
    data = {}

# Find latest version
latest_version = 0
for key in data.keys():
    try:
        version_num = int(key.split("_")[-1])
        if version_num > latest_version:
            latest_version = version_num
    except (ValueError, IndexError):
        continue

new_version = latest_version + 1
data[f"data_{new_version}"] = {
    "version": new_version_str,
    "description": args.commit_msg,
    "git-hash": args.commit_id,
}

# Write updated data back to file
with open("data-experiments.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

# Update version.py with the new minor version
with open("src/data_pipeline/version.py", "w") as version_file:
    version_file.write(f"MAJOR_VERSION = {MAJOR_VERSION}\n")
    version_file.write(f"MINOR_VERSION = {new_minor_version}\n")
    version_file.write(f"PATCH_VERSION = {PATCH_VERSION}\n")
    version_file.write(f"TWEAK_VERSION = {TWEAK_VERSION}\n")
