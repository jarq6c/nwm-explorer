"""Main."""
from pathlib import Path

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import download_nwm, ModelConfiguration, scan_nwm

if __name__ == "__main__":
    # Data directory
    root = Path("./data")

    # Load routelink
    rl = download_routelink(
        file_path=root / "routelink.parquet"
    ).select(
        ["nwm_feature_id", "domain"]
    ).collect()

    # Download and process NWM output
    download_nwm(
        start=pd.Timestamp("2025-04-01"),
        end=pd.Timestamp("2025-04-04"),
        root=root,
        routelink=rl,
        jobs=18
    )

    # Check data
    result = scan_nwm(root)
    print(
        result.filter(
            pl.col("configuration") == ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA,
            pl.col("nwm_feature_id") == 75000700032122
        ).select(
            ["value_time", "predicted_cfs"]
        ).sort(
            by=pl.col("value_time")
        ).collect()
    )
