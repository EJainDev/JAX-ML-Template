import argparse

import yaml

from src.data_pipeline.version import MAJOR_VERSION as DATA_MAJOR_VERSION
from src.data_pipeline.version import MINOR_VERSION as DATA_MINOR_VERSION
from src.data_pipeline.version import PATCH_VERSION as DATA_PATCH_VERSION
from src.data_pipeline.version import TWEAK_VERSION as DATA_TWEAK_VERSION
from src.training_engine.flax_training_engine import \
    MAJOR_VERSION as TE_MAJOR_VERSION
from src.training_engine.flax_training_engine import \
    MINOR_VERSION as TE_MINOR_VERSION
from src.training_engine.flax_training_engine import \
    PATCH_VERSION as TE_PATCH_VERSION
from src.training_engine.flax_training_engine import \
    TWEAK_VERSION as TE_TWEAK_VERSION
from src.version import (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION,
                         TWEAK_VERSION)

parser = argparse.ArgumentParser(
    description="Update experiment version in YAML file")
parser.add_argument("--commit_id", type=str,
                    help="The git commit id", required=True)
parser.add_argument(
    "--commit_msg", type=str, help="The git commit message", required=True
)
args = parser.parse_args()

# Increment the minor version from version.py
NEW_MINOR_VERSION = MINOR_VERSION + 1

# Read the experiments YAML file
with open("experiments.yaml", "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

# Handle empty file
if data is None:
    data = {}

# Find latest experiment version
LATEST_VERSION = 0
for key in data.keys():
    try:
        version_num = int(key.split("_")[-1])
        LATEST_VERSION = max(LATEST_VERSION, version_num)
    except (ValueError, IndexError):
        continue

# 2) Use the description as the commit message (argument)
# 3) Use the git hash as the commit id (argument)
new_version = LATEST_VERSION + 1
data[f"experiment_{new_version}"] = {
    "version": f"{MAJOR_VERSION}.{NEW_MINOR_VERSION}.{PATCH_VERSION}.{TWEAK_VERSION}",
    "data-version":
        f"{DATA_MAJOR_VERSION}.{DATA_MINOR_VERSION}.{DATA_PATCH_VERSION}.{DATA_TWEAK_VERSION}",
    "training-engine-version":
        f"{TE_MAJOR_VERSION}.{TE_MINOR_VERSION}.{TE_PATCH_VERSION}.{TE_TWEAK_VERSION}",
    "description": args.commit_msg,
    "git-hash": args.commit_id,
}

# Write updated data back to file
with open("experiments.yaml", "w", encoding="utf-8") as file:
    yaml.dump(data, file, default_flow_style=False, sort_keys=False)

# Update version.py with the new minor version
with open("src/version.py", "w", encoding="utf-8") as version_file:
    version_file.write(f"MAJOR_VERSION = {MAJOR_VERSION}\n")
    version_file.write(f"MINOR_VERSION = {NEW_MINOR_VERSION}\n")
    version_file.write(f"PATCH_VERSION = {PATCH_VERSION}\n")
    version_file.write(f"TWEAK_VERSION = {TWEAK_VERSION}\n")
