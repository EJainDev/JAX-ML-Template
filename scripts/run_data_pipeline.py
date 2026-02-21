import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "src")


def run_command(cmd, project_root):
    """Run a shell cmd silently."""
    try:
        subprocess.run(
            cmd,
            shell=True,
            check=True,
            cwd=project_root,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(e)
        return False


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.resolve()
    # Get the venv Python executable
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"

    # Run pipeline steps
    steps = [
        f"{venv_python} -m src.data_pipeline.load_raw",
        f"{venv_python} -m src.data_pipeline.process_data",
    ]

    for command in steps:
        run_command(command, PROJECT_ROOT)
