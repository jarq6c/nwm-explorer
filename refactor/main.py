"""Main."""
from pathlib import Path

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import scan_nwm, ModelConfiguration
from modules.usgs import scan_usgs, load_usgs, lookup_site_state_code

if __name__ == "__main__":
    root = Path("/ised/nwm_explorer_data")

    print(lookup_site_state_code(root, "02146470"))
    print(lookup_site_state_code(root, "02146470"))
    quit()
    from time import perf_counter, sleep

    for _ in range(3):
        start = perf_counter()
        for month in [4, 5, 6]:
            df = load_usgs(
                root,
                "nc",
                2024,
                month,
                True
            ).filter(pl.col("usgs_site_code") == "02146470")
        duration = perf_counter() - start
        print(duration)
    sleep(5)
    quit()
    # Data directory
    root = Path("./data")

    # Load routelink
    rl = download_routelink(
        root
    ).select(
        ["nwm_feature_id", "usgs_site_code", "domain"]
    ).collect()

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

    # Prepare observations
    observations = scan_usgs(root
        ).filter(
            pl.col("usgs_site_code").is_in(rl["usgs_site_code"].implode())
        ).select(
            ["usgs_site_code", "value_time", "observed_cfs"]
        ).unique(
            ["usgs_site_code", "value_time"]
        ).sort(
            ["usgs_site_code", "value_time"]
        ).group_by_dynamic(
            "value_time",
            every="1d",
            group_by="usgs_site_code"
        ).agg(
            pl.col("observed_cfs").max()
        ).with_columns(
            nwm_feature_id=pl.col("usgs_site_code").replace_strict(
                rl["usgs_site_code"].implode(),
                rl["nwm_feature_id"].implode()
            )
        )

    predictions = scan_nwm(root
        ).filter(
            pl.col("configuration") == ModelConfiguration.MEDIUM_RANGE_MEM_1
        ).select(
            ["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
        ).with_columns(
            ((pl.col("value_time") - pl.col("reference_time")) /
                pl.duration(hours=24)).round().cast(pl.Int32).alias("lead_time_hours_min")
        ).sort(
            ["nwm_feature_id", "lead_time_hours_min", "value_time"]
        ).group_by_dynamic(
            "value_time",
            every="1d",
            group_by=["nwm_feature_id", "lead_time_hours_min"]
        ).agg(
            pl.col("predicted_cfs").max(),
            # pl.col("value_time").min().alias("value_time_min"),
            # pl.col("value_time").max().alias("value_time_max"),
            # pl.col("reference_time").min().alias("reference_time_min"),
            # pl.col("reference_time").max().alias("reference_time_max")
        )
    print(observations.collect().estimated_size("mb"))
