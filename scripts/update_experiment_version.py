import yaml
import argparse

parser = argparse.ArgumentParser(description="Update data version in YAML file")
parser.add_argument("--commit_id", type=str, help="The git commit id", required=True)
parser.add_argument(
    "--commit_msg", type=str, help="The git commit message", required=True
)
args = parser.parse_args()

# Read the YAML file
with open("experiments.yaml", "r") as file:
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

# Get current version and increment
if latest_version > 0 and f"experiment_{latest_version}" in data:
    version = data[f"experiment_{latest_version}"]["version"]
    version_keys = version.split(".")
    new_version_str = f"{version_keys[0]}.{int(version_keys[1]) + 1}.{version_keys[2]}.{version_keys[3]}"
else:
    new_version_str = "0.1.0.0"

new_version = latest_version + 1
data[f"experiment_{new_version}"] = {
    "version": new_version_str,
    "data-version": data[f"experiment_{latest_version}"]["data-version"],
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
