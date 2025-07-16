"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor

import polars as pl
import pandas as pd

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.data.nwm import generate_reference_dates, build_nwm_file_details, NWM_URL_BUILDERS
from nwm_explorer.data.usgs import get_usgs_reader

OBSERVATION_RESAMPLING: dict[ModelDomain, tuple[str]] = {
    ModelDomain.alaska: ("1d", "5h"),
    ModelDomain.conus: ("1d", "6h"),
    ModelDomain.hawaii: ("1d", "6h"),
    ModelDomain.puertorico: ("1d", "6h")
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

    # File details
    logger.info("Generating file details")
    file_details = build_nwm_file_details(root, reference_dates)

    # Download
    logger.info("Pairing NWM data")
    for fd in file_details:
        if fd.path.exists():
            ofile = Path(fd.path.parent) / fd.path.name.replace("streamflow", "pairs")
            if ofile.exists():
                logger.info(f"Found {ofile}")
                continue
            logger.info(f"Building {ofile}")
            logger.info(f"Loading {fd.path}")
            sim = pl.read_parquet(fd.path)
            first = sim["value_time"].min()
            last = sim["value_time"].max()

            logger.info("Loading observations")
            obs = get_usgs_reader(root, fd.domain, first, last).with_columns(
                pl.col("observed").cast(pl.Float32)
            ).collect()
            xwalk = crosswalk[fd.domain]

            logger.info(f"Resampling")
            if fd.configuration in PREDICTION_RESAMPLING:
                sampling_duration = PREDICTION_RESAMPLING[fd.configuration][0]
                resampling_frequency = PREDICTION_RESAMPLING[fd.configuration][1]
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
                obs = obs.sort(
                    ("usgs_site_code", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every=resampling_frequency,
                    group_by="usgs_site_code"
                ).agg(
                    pl.col("observed").max()
                )
            else:
                # NOTE This will result in two simulation values per
                #  reference day. Handle this before computing metrics (max).
                sim = sim.sort(
                    ("nwm_feature_id", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every="1d",
                    group_by="nwm_feature_id"
                ).agg(
                    pl.col("predicted").max()
                )
                obs = obs.sort(
                    ("usgs_site_code", "value_time")
                ).group_by_dynamic(
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
            pairs.write_parquet(ofile)

def build_pairs_filepath(
    root: Path,
    domain: ModelDomain,
    configuration: ModelConfiguration,
    reference_date: pd.Timestamp
    ) -> Path:
    date_string = reference_date.strftime("nwm.%Y%m%d")
    return root / "parquet" / domain / date_string / f"{configuration}_pairs_cfs.parquet"

def get_pairs_reader(
    root: Path,
    domain: ModelDomain,
    configuration: ModelConfiguration,
    reference_dates: list[pd.Timestamp]
    ) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Get file path
    logger.info(f"Scanning {domain} {configuration} {reference_dates[0]} to {reference_dates[-1]}")
    file_paths = [build_pairs_filepath(root, domain, configuration, rd) for rd in reference_dates]
    return pl.scan_parquet([fp for fp in file_paths if fp.exists()])

def get_pairs_readers(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path
    ) -> dict[tuple[ModelDomain, ModelConfiguration], pl.LazyFrame]:
    """Returns mapping from ModelDomain to polars.LazyFrame."""
    # Generate reference dates
    reference_dates = generate_reference_dates(startDT, endDT)
    return {(d, c): get_pairs_reader(root, d, c, reference_dates) for d, c in NWM_URL_BUILDERS}

def compute_metrics(data: pd.DataFrame) -> dict[str, float]:
    return {
        "nwm_feature_id": data["nwm_feature_id"].iloc[0],
        "usgs_site_code": data["usgs_site_code"].iloc[0],
        "nash_sutcliffe_efficiency": 1.5
    }

def run_standard_evaluation(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Pair data
    logger.info("Checking for pairs")
    generate_pairs(
        startDT,
        endDT,
        root,
        routelinks
    )

    # Setup pool
    logger.info("Setup compute resources")
    pool = ProcessPoolExecutor(max_workers=jobs)

    # Scan
    logger.info("Scan pairs")
    pairs = get_pairs_readers(startDT, endDT, root)
    for (d, c), data in pairs.items():
        logger.info(f"Evaluating {d} {c}")
        # Handle simulations
        if c not in PREDICTION_RESAMPLING:
            # Resolve duplicate predictions
            data = data.sort(
                    ("nwm_feature_id", "value_time")
                ).group_by_dynamic(
                    "value_time",
                    every="1d",
                    group_by="nwm_feature_id"
                ).agg(
                    pl.col("predicted").max(),
                    pl.col("observed").max(),
                    pl.col("usgs_site_code").first()
                ).with_columns(pl.col("usgs_site_code").cast(pl.String)).collect()
            
            # Group by feature id
            crosswalk = data.select("nwm_feature_id").unique("nwm_feature_id")
            features = crosswalk["nwm_feature_id"].to_numpy()
            dataframes = [data.filter(pl.col("nwm_feature_id") == fid).to_pandas() for fid in features]

            # Evaluate
            chunk_size = max(1, len(dataframes) // jobs)
            metrics = pd.DataFrame.from_records(pool.map(compute_metrics, dataframes, chunksize=chunk_size))
            print(metrics.head())

    # Clean-up
    logger.info("Cleaning up compute resources")
    pool.shutdown()
