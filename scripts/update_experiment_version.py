import yaml
import argparse
import sys

sys.path.insert(0, "src")
from flax_training_engine import (
    MAJOR_VERSION as TE_MAJOR_VERSION,
    MINOR_VERSION as TE_MINOR_VERSION,
    PATCH_VERSION as TE_PATCH_VERSION,
    TWEAK_VERSION as TE_TWEAK_VERSION,
)
from src.version import (
    MAJOR_VERSION,
    MINOR_VERSION,
    PATCH_VERSION,
    TWEAK_VERSION,
)
from src.data_pipeline.version import (
    MAJOR_VERSION as DATA_MAJOR_VERSION,
    MINOR_VERSION as DATA_MINOR_VERSION,
    PATCH_VERSION as DATA_PATCH_VERSION,
    TWEAK_VERSION as DATA_TWEAK_VERSION,
)

parser = argparse.ArgumentParser(description="Update experiment version in YAML file")
parser.add_argument("--commit_id", type=str, help="The git commit id", required=True)
parser.add_argument(
    "--commit_msg", type=str, help="The git commit message", required=True
)
args = parser.parse_args()

# Increment the minor version from version.py
new_minor_version = MINOR_VERSION + 1

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

# 2) Use the description as the commit message (argument)
# 3) Use the git hash as the commit id (argument)
new_version = latest_version + 1
data[f"experiment_{new_version}"] = {
    "version": f"{MAJOR_VERSION}.{new_minor_version}.{PATCH_VERSION}.{TWEAK_VERSION}",
    "data-version": f"{DATA_MAJOR_VERSION}.{DATA_MINOR_VERSION}.{DATA_PATCH_VERSION}.{DATA_TWEAK_VERSION}",
    "training-engine-version": f"{TE_MAJOR_VERSION}.{TE_MINOR_VERSION}.{TE_PATCH_VERSION}.{TE_TWEAK_VERSION}",
    "description": args.commit_msg,
    "git-hash": args.commit_id,
}

# Write updated data back to file
with open("experiments.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

# Update version.py with the new minor version
with open("src/version.py", "w") as version_file:
    version_file.write(f"MAJOR_VERSION = {MAJOR_VERSION}\n")
    version_file.write(f"MINOR_VERSION = {new_minor_version}\n")
    version_file.write(f"PATCH_VERSION = {PATCH_VERSION}\n")
    version_file.write(f"TWEAK_VERSION = {TWEAK_VERSION}\n")
