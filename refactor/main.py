"""Main."""
from pathlib import Path

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import download_nwm, ModelConfiguration, scan_nwm
from modules.usgs import download_usgs, scan_usgs, download_site_table, scan_site_table

if __name__ == "__main__":
    # Data directory
    root = Path("./data")

    # Download site table
    download_site_table(root)
    site_table = scan_site_table(root).collect()
    print(site_table)

    # # Load routelink
    # rl = download_routelink(
    #     file_path=root / "routelink.parquet"
    # ).select(
    #     ["nwm_feature_id", "domain"]
    # ).collect()

    # # Download and process NWM output
    # download_nwm(
    #     start=pd.Timestamp("2025-04-01"),
    #     end=pd.Timestamp("2025-04-05"),
    #     root=root,
    #     routelink=rl,
    #     jobs=18
    # )

    # # Extract required dates of observations
    # predictions = scan_nwm(root)
    # date_range = predictions.select(
    #     pl.col("value_time").min().alias("start"),
    #     pl.col("value_time").max().alias("end")
    #     ).collect()
    # start = date_range["start"].item(0).strftime("%Y-%m-%d")
    # end = date_range["end"].item(0).strftime("%Y-%m-%d")

    # # Download and process USGS streamflow
    # download_usgs(
    #     start=pd.Timestamp(start),
    #     end=pd.Timestamp(end),
    #     root=root
    # )

    # # Check observations
    # observations = scan_usgs(root)

    # print(observations.collect())
