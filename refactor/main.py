"""Methods to evaluate pairs."""
from pathlib import Path

import polars as pl

from modules.nwm import ModelConfiguration
from modules.pairs import scan_pairs

if __name__ == "__main__":
    mroot = Path("/ised/nwm_explorer_data")

    mpairs = scan_pairs(mroot).filter(
        pl.col("configuration") == ModelConfiguration.MEDIUM_RANGE_MEM_1,
        pl.col("year") == 2023,
        pl.col("month").is_in([10, 11, 12]),
        # pl.col("nwm_feature_id") == 3109
    ).select([
        "nwm_feature_id",
        "reference_time",
        "value_time",
        "predicted_cfs_max",
        "observed_cfs_max"
    ]).collect().with_columns(
        lead_time_hours_min=(
            pl.col("value_time").sub(pl.col("reference_time")) / pl.duration(hours=24)
        ).floor().cast(pl.Int32).mul(24)
    ).drop(["reference_time", "value_time"])

    print(mpairs.estimated_size("mb"))
