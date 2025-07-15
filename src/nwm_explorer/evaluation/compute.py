"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect

import polars as pl

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration

OBSERVATION_RESAMPLING: dict[ModelDomain, list[str]] = {
    ModelDomain.alaska: ["1d", "5h"],
    ModelDomain.conus: ["1d", "6h"],
    ModelDomain.hawaii: ["1d", "6h"],
    ModelDomain.puertorico: ["1d", "6h"]
}
"""Mapping used to resample observations."""

PREDICTION_RESAMPLING: dict[ModelConfiguration, tuple[pl.Duration, str]] = {
    ModelConfiguration.medium_range_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_mem1: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_blend_alaska: (pl.duration(hours=24), "1d"),
    ModelConfiguration.medium_range_alaska_no_da: (pl.duration(hours=24), "1d"),
    ModelConfiguration.short_range: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_alaska: (pl.duration(hours=5), "5h"),
    ModelConfiguration.short_range_hawaii: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_hawaii_no_da: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico: (pl.duration(hours=6), "6h"),
    ModelConfiguration.short_range_puertorico_no_da: (pl.duration(hours=6), "6h")
}
"""Mapping used for computing lead time and sampling frequency."""

def compute_metrics(
        predictions: dict[tuple[ModelDomain, ModelConfiguration], pl.LazyFrame],
        observations: dict[ModelDomain, pl.LazyFrame],
        routelinks: dict[ModelDomain, pl.LazyFrame],
        root: Path
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Resampling observations")
    obs: dict[ModelDomain, dict[str, pl.DataFrame]] = {}
    for d, o in observations.items():
        dataframes = {}
        for resampling_frequency in OBSERVATION_RESAMPLING[d]:
            dataframes[resampling_frequency] = o.sort(
                    ("usgs_site_code", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every=resampling_frequency,
                    group_by="usgs_site_code"
                ).agg(
                    pl.col("observed").max()
                ).with_columns(
                    pl.col("usgs_site_code").cast(pl.String)
                )
        obs[d] = dataframes

    logger.info("Computing metrics")
    for (d, c), pred in predictions.items():
        logger.info(f"Pairing {d} {c}")
        obs = observations[d]
        rl = routelinks[d].select(["usgs_site_code", "nwm_feature_id"])
        print(rl.head().collect())
        break
