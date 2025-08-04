"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect

import polars as pl
import pandas as pd

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.data.nwm import generate_reference_dates, get_nwm_readers, NWM_URL_BUILDERS, build_nwm_filepath
from nwm_explorer.data.usgs import get_usgs_readers

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

def build_pairs_filepath(
    root: Path,
    domain: ModelDomain,
    configuration: ModelConfiguration,
    reference_date: pd.Timestamp
    ) -> Path:
    date_string = reference_date.strftime("pairs.%Y%m%d")
    return root / "parquet" / domain / date_string / f"{configuration}_pairs_cfs.parquet"

def generate_pairs(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame]
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate reference dates
    logger.info("Generating reference dates")
    reference_dates = generate_reference_dates(startDT, endDT)

    # Read routelinks
    logger.info("Reading routelinks")
    crosswalk = {d: df.select(["usgs_site_code", "nwm_feature_id"]).collect() for d, df in routelinks.items()}

    # Scan NWM data
    logger.info("Scanning model output")
    predictions = get_nwm_readers(startDT, endDT, root)

    # Determine date range for observations
    first = startDT
    last = endDT
    for df in predictions.values():
        first = min(first, df.select("value_time").min().collect().item(0, 0))
        last = max(last, df.select("value_time").max().collect().item(0, 0))

    # Scan USGS data
    logger.info("Scanning observations")
    observations = get_usgs_readers(
        pd.Timestamp(first),
        pd.Timestamp(last),
        root
        )

    # Process files, one at a time
    logger.info("Pairing model output")
    for (d, c) in NWM_URL_BUILDERS.keys():
        for rd in reference_dates:
            # Build output file path
            ofile = build_pairs_filepath(root, d, c, rd)
            logger.info(f"Building {ofile}")

            # Do not overwrite
            if ofile.exists():
                logger.info(f"Found {ofile}, skipping")
                continue

            # Handling file system details
            ofile.parent.mkdir(exist_ok=True, parents=True)
            ifile = build_nwm_filepath(root, d, c, rd)

            # Load model output
            logger.info(f"Loading {ifile}")
            sim = pl.read_parquet(ifile)

            # Set crosswalk
            xwalk = crosswalk[d]

            # Load corresponding observations
            logger.info("Loading observations")
            first = sim["value_time"].min()
            last = sim["value_time"].max()
            obs = observations[d].filter(
                pl.col("value_time") >= first,
                pl.col("value_time") <= last,
                pl.col("usgs_site_code").is_in(xwalk["usgs_site_code"])
            ).select(
                ["value_time", "usgs_site_code", "observed"]
            ).unique(
                subset=["value_time", "usgs_site_code"]
            ).sort(
                ("usgs_site_code", "value_time")
            ).collect()

            # Resample
            logger.info("Resampling")
            if c in PREDICTION_RESAMPLING:
                sampling_duration = PREDICTION_RESAMPLING[c][0]
                resampling_frequency = PREDICTION_RESAMPLING[c][1]
                hours = sampling_duration / pl.duration(hours=1)
                sim = sim.sort(
                    ("nwm_feature_id", "reference_time", "value_time")
                ).with_columns(
                    ((pl.col("value_time").sub(
                        pl.col("reference_time")
                        ) / sampling_duration).floor() *
                            hours).cast(pl.Int32).alias("lead_time_hours_min")
                ).group_by_dynamic(
                    "value_time",
                    every=resampling_frequency,
                    group_by=("nwm_feature_id", "reference_time")
                ).agg(
                    pl.col("predicted").max(),
                    pl.col("lead_time_hours_min").min()
                )
                obs = obs.group_by_dynamic(
                    "value_time",
                    every=resampling_frequency,
                    group_by="usgs_site_code"
                ).agg(
                    pl.col("observed").max()
                )
            else:
                # NOTE This will result in two simulation values per
                #  reference day. Handle this in reader methods.
                sim = sim.sort(
                    ("nwm_feature_id", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every="1d",
                    group_by="nwm_feature_id"
                ).agg(
                    pl.col("predicted").max()
                )
                obs = obs.group_by_dynamic(
                    "value_time",
                    every="1d",
                    group_by="usgs_site_code"
                ).agg(
                    pl.col("observed").max()
                )
            
            obs = obs.with_columns(
                nwm_feature_id=pl.col("usgs_site_code").replace_strict(
                    xwalk["usgs_site_code"], xwalk["nwm_feature_id"])
                )
            pairs = sim.join(obs, on=["nwm_feature_id", "value_time"],
                how="left").drop_nulls()
            
            logger.info(f"Saving {ofile}")
            # pairs.write_parquet(ofile)
            print(pairs.head())

# def get_pairs_reader(
#     root: Path,
#     domain: ModelDomain,
#     configuration: ModelConfiguration,
#     reference_dates: list[pd.Timestamp]
#     ) -> pl.LazyFrame:
#     # Get logger
#     name = __loader__.name + "." + inspect.currentframe().f_code.co_name
#     logger = get_logger(name)
    
#     # Get file path
#     logger.info(f"Scanning {domain} {configuration} {reference_dates[0]} to {reference_dates[-1]}")
#     file_paths = [build_pairs_filepath(root, domain, configuration, rd) for rd in reference_dates]
#     return pl.scan_parquet([fp for fp in file_paths if fp.exists()])

# def get_pairs_readers(
#     startDT: pd.Timestamp,
#     endDT: pd.Timestamp,
#     root: Path
#     ) -> dict[tuple[ModelDomain, ModelConfiguration], pl.LazyFrame]:
#     """Returns mapping from ModelDomain to polars.LazyFrame."""
#     # Generate reference dates
#     reference_dates = generate_reference_dates(startDT, endDT)
#     return {(d, c): get_pairs_reader(root, d, c, reference_dates) for d, c in NWM_URL_BUILDERS}
