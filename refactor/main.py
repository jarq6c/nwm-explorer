"""Methods to evaluate pairs."""
from pathlib import Path

import polars as pl
import pandas as pd

from modules.nwm import ModelConfiguration
from modules.routelink import download_routelink
from modules.pairs import scan_pairs

def main(
        root: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lead_time_scale: int = 24,
        maximum_processes: int = 1,
        chunk_size: int = 1
        ) -> None:
    """Main."""
    # Load feature list
    features = download_routelink(root).select(
        "nwm_feature_id").collect()["nwm_feature_id"].to_list()

    # Load data
    dataframes = []
    for m in pd.date_range(start_time, end_time, freq="1MS"):
        dataframes.append(
            scan_pairs(root, cache=True).filter(
                pl.col("configuration") == ModelConfiguration.MEDIUM_RANGE_MEM_1,
                pl.col("year") == m.year,
                pl.col("month") == m.month,
                pl.col("reference_time") >= start_time,
                pl.col("reference_time") <= end_time,
                # pl.col("nwm_feature_id").is_in(features[:100])
            ).collect()
        )
    # TODO Group by NWM Feature ID and Lead time, every 1D of value_time

    data = pl.concat(dataframes).filter(
        pl.col("nwm_feature_id") == 3109,
        pl.col("value_time") == pd.Timestamp("2023-10-12")
    )
    data.write_csv("test_batch.csv")

if __name__ == "__main__":
    main(
        Path("/ised/nwm_explorer_data"),
        pd.Timestamp("2023-10-01"),
        pd.Timestamp("2023-12-31")
    )
