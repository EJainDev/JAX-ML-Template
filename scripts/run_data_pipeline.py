import subprocess
import sys
from pathlib import Path


def run_command(command):
    """Run a shell command silently."""
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            cwd=PROJECT_ROOT,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        return False


def main():
    """Execute the data pipeline."""
    # Get the venv Python executable
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    # Run pipeline steps
    steps = [
        f"{venv_python} -m src.data_pipeline.load_raw",
        f"{venv_python} -m src.data_pipeline.process_data",
    ]

    for command in steps:
        if not run_command(command):
            return 1

    return 0


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    sys.exit(main())
