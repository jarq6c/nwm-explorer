"""Methods to evaluate pairs."""
from pathlib import Path
import inspect

import polars as pl
import pandas as pd

from modules.nwm import ModelConfiguration
from modules.pairs import scan_pairs, GROUP_SPECIFICATIONS
from modules.logger import get_logger

def generate_prediction_pools(
        root: Path,
        configuration: ModelConfiguration,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        lead_time_interval: int
        ) -> pl.DataFrame:
    """
    Load and group forecast pairs into lead time pools.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        NWM Model configuration.
    start_time: pandas.Timestamp
        First value time.
    end_time: pandas.Timestamp
        Last value time.
    lead_time_interval: int
        Lead time scale to aggregate over in hours.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Load data
    logger.info("Loading pairs")
    dataframes = []
    for m in pd.date_range(start_time, end_time, freq="1MS"):
        dataframes.append(
            scan_pairs(root, cache=True).filter(
                pl.col("configuration") == configuration,
                pl.col("year") == m.year,
                pl.col("month") == m.month,
                pl.col("reference_time") >= start_time,
                pl.col("reference_time") <= end_time
            ).collect()
        )

    # Compute lead time and pool pairs
    logger.info("Pooling pairs")
    return pl.concat(
        dataframes
    ).with_columns(
        lead_time_pool=(pl.col("predicted_value_time_min").sub(pl.col("reference_time")) /
            pl.duration(hours=lead_time_interval)).floor().cast(pl.Int32)
    ).sort([
        "configuration",
        "nwm_feature_id",
        "lead_time_pool",
        "value_time"]
    ).group_by_dynamic(
        "value_time",
        every=f"{lead_time_interval}h",
        group_by=["configuration", "nwm_feature_id", "lead_time_pool"]
    ).agg(
        pl.col("predicted_cfs_min").min(),
        pl.col("predicted_cfs_median").median(),
        pl.col("predicted_cfs_max").max(),
        pl.col("predicted_value_time_min").min(),
        pl.col("predicted_value_time_max").max(),
        pl.col("reference_time").min().alias("reference_time_min"),
        pl.col("reference_time").max().alias("reference_time_max"),
        pl.col("usgs_site_code").first(),
        pl.col("observed_cfs_min").min(),
        pl.col("observed_cfs_median").median(),
        pl.col("observed_cfs_max").max(),
        pl.col("observed_value_time_min").min(),
        pl.col("observed_value_time_max").max()
    )

def main() -> None:
    """Main."""
    # Process each configuration
    for config, specs in GROUP_SPECIFICATIONS.items():
        print(config)
        df = generate_prediction_pools(
            root=Path("/ised/nwm_explorer_data"),
            configuration=config,
            start_time=pd.Timestamp("2023-10-01"),
            end_time=pd.Timestamp("2023-12-31"),
            lead_time_interval=specs.window_interval
        )
        print(df.estimated_size("mb"))
        break

if __name__ == "__main__":
    main()
