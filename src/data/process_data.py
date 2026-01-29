import os
import joblib
import polars as pl

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..config import (
    PROCESSED_DATA_DIR,
    PROCESSED_DATA_FILE,
    RAW_DATA_FILE,
    SCALER_PATH,
    TRAIN_VAL_TEST_SPLIT,
)


def process_data(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    return df


if __name__ == "__main__":
    df: pl.LazyFrame = pl.scan_csv(RAW_DATA_FILE, try_parse_dates=True)
    temp_df: pl.LazyFrame | pl.DataFrame = process_data(df)

    if type(temp_df) == pl.LazyFrame:
        processed_df: pl.DataFrame = temp_df.collect()
    elif type(temp_df) == pl.DataFrame:
        processed_df = temp_df
    else:
        raise TypeError(
            "Processed data is not the expected type (pl.LazyFrame | pl.DataFrame)"
        )

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    scaler = StandardScaler()
    scaler.fit(
        train_test_split(processed_df.to_numpy(), train_size=TRAIN_VAL_TEST_SPLIT[0])[0]
    )
    joblib.dump(scaler, SCALER_PATH)

    processed_df.write_parquet(PROCESSED_DATA_FILE)
