from pathlib import Path

from ..config import RAW_DATA_DIR, RAW_DATA_FILE


def load_raw_data(output_path: str | Path) -> Path:
    return RAW_DATA_FILE


if __name__ == "__main__":
    print(f"Raw dataset saved to {load_raw_data(output_path=RAW_DATA_DIR)}")
