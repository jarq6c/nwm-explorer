"""Main."""
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
import polars as pl

from modules.routelink import download_routelink
from modules.nwm import scan_nwm, ModelConfiguration
from modules.usgs import scan_usgs

@dataclass
class NWMGroupSpecification:
    """
    A dataclass that holds specifications for time-based polars groupby
    operations.
    """
    index_column: str = "value_time"
    group_by_columns: list[str] | None = None
    window_interval: str = "1d"

    def __post_init__(self) -> None:
        if self.group_by_columns is None:
            self.group_by_columns = ["nwm_feature_id", "reference_time"]

    @property
    def select_columns(self) -> list[str]:
        """A list of columns to select."""
        return self.group_by_columns + [self.index_column, "predicted_cfs"]

    @property
    def sort_columns(self) -> list[str]:
        """A list of columns to sort by."""
        return self.group_by_columns + [self.index_column]

GROUP_SPECIFICATIONS: dict[ModelConfiguration, NWMGroupSpecification] = {
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_MEM_1: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_BLEND: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA: NWMGroupSpecification(),
    ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE_ALASKA: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE_HAWAII: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO: NWMGroupSpecification(),
    ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA: NWMGroupSpecification(),
}
"""Mapping from ModelConfiguration to group-by specifications."""

if __name__ == "__main__":
    root = Path("/ised/nwm_explorer_data")

    # Predictions
    predictions = scan_nwm(root)
    date_range = pd.date_range(start="2023-10-01", end="2025-09-30", freq="1MS")
    for configuration in list(ModelConfiguration):
        for reference_date in date_range:
            print(configuration, reference_date)
            pairs = predictions.filter(
                pl.col("configuration") == configuration,
                pl.col("year") == reference_date.year,
                pl.col("month") == reference_date.month
            ).select(
                ["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
            ).collect().sort(
                ["nwm_feature_id", "reference_time", "value_time"]
            ).group_by_dynamic(
                "value_time",
                every="1d",
                group_by=["nwm_feature_id", "reference_time"]
            ).agg(
                pl.col("predicted_cfs").min().alias("predicted_cfs_min"),
                pl.col("predicted_cfs").max().alias("predicted_cfs_max"),
                # pl.col("value_time").min().alias("value_time_min"),
                # pl.col("value_time").max().alias("value_time_max")
            )
            print(pairs)
            break
        break

    # Load routelink
    # rl = download_routelink(
    #     root
    # ).select(
    #     ["nwm_feature_id", "usgs_site_code", "domain"]
    # ).collect()

    # Prepare observations
    # observations = scan_usgs(root
    #     ).filter(
    #         # pl.col("usgs_site_code").is_in(rl["usgs_site_code"].implode())
    #         pl.col("usgs_site_code") == "02146470"
    #     ).select(
    #         ["usgs_site_code", "value_time", "observed_cfs"]
    #     ).unique(
    #         ["usgs_site_code", "value_time"]
    #     ).sort(
    #         ["usgs_site_code", "value_time"]
    #     ).group_by_dynamic(
    #         "value_time",
    #         every="1d",
    #         group_by="usgs_site_code"
    #     ).agg(
    #         pl.col("observed_cfs").max()
    #     ).with_columns(
    #         nwm_feature_id=pl.col("usgs_site_code").replace_strict(
    #             rl["usgs_site_code"].implode(),
    #             rl["nwm_feature_id"].implode()
    #         )
    #     )
    # print(observations.head().collect())

    # predictions = scan_nwm(root
    #     ).filter(
    #         pl.col("configuration") == ModelConfiguration.MEDIUM_RANGE_MEM_1
    #     ).select(
    #         ["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    #     ).with_columns(
    #         ((pl.col("value_time") - pl.col("reference_time")) /
    #             pl.duration(hours=24)).round().cast(pl.Int32).alias("lead_time_hours_min")
    #     ).sort(
    #         ["nwm_feature_id", "lead_time_hours_min", "value_time"]
    #     ).group_by_dynamic(
    #         "value_time",
    #         every="1d",
    #         group_by=["nwm_feature_id", "lead_time_hours_min"]
    #     ).agg(
    #         pl.col("predicted_cfs").max(),
    #         # pl.col("value_time").min().alias("value_time_min"),
    #         # pl.col("value_time").max().alias("value_time_max"),
    #         # pl.col("reference_time").min().alias("reference_time_min"),
    #         # pl.col("reference_time").max().alias("reference_time_max")
    #     )
    # print(observations.collect().estimated_size("mb"))
