"""Main."""
from pathlib import Path

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import download_nwm, ModelConfiguration

if __name__ == "__main__":
    # Load routelink
    rl = download_routelink(
        file_path=Path("./data/routelink.parquet")
    ).select(
        ["nwm_feature_id", "domain"]
    ).collect()

    # Download and process NWM output
    download_nwm(
        start=pd.Timestamp("2025-04-01"),
        end=pd.Timestamp("2025-04-03"),
        root=Path("./data"),
        routelink=rl,
        jobs=18
    )

    # Check data
    result = pl.scan_parquet(
        "data/nwm/",
        hive_schema={
            "configuration": pl.Enum(ModelConfiguration),
            "year": pl.Int32,
            "month": pl.Int32
        }
    )
    print(
        result.filter(
            pl.col("configuration") == ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA,
            pl.col("nwm_feature_id") == 75000700032122
        ).select(
            ["value_time", "predicted_cfs"]
        ).collect()
    )
