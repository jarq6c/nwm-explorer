"""Compute standard evaluation metrics."""
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

import polars as pl
import pandas as pd
from pydantic import BaseModel

from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.data.nwm import generate_reference_dates, build_nwm_file_details, NWM_URL_BUILDERS
from nwm_explorer.data.usgs import get_usgs_reader
from nwm_explorer.evaluation.metrics import bootstrap_metrics

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

def get_evaluation_reader(
    domain: ModelDomain,
    configuration: ModelConfiguration,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path
    ) -> pl.LazyFrame | None:
    start_string = startDT.strftime("%Y%m%d")
    end_string = endDT.strftime("%Y%m%d")
    odir = root / f"parquet/{domain}/evaluations"
    odir.mkdir(exist_ok=True)
    ofile = odir / f"{configuration}_{start_string}_{end_string}.parquet"
    return pl.scan_parquet(ofile)

def get_evaluation_readers(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path
    ) -> dict[tuple[ModelDomain, ModelConfiguration], pl.LazyFrame]:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Scan files
    evaluations = {}
    start_string = startDT.strftime("%Y%m%d")
    end_string = endDT.strftime("%Y%m%d")
    for (d, c), _ in NWM_URL_BUILDERS.items():
        odir = root / f"parquet/{d}/evaluations"
        odir.mkdir(exist_ok=True)
        ofile = odir / f"{c}_{start_string}_{end_string}.parquet"
        if ofile.exists():
            logger.info(f"Found {ofile}")
            evaluations[(d, c)] = pl.scan_parquet(ofile)

class EvaluationSpec(BaseModel):
    startDT: datetime
    endDT: datetime
    directory: Path
    files: dict[ModelDomain, dict[ModelConfiguration, Path]]

class EvaluationRegistry(BaseModel):
    evaluations: dict[str, EvaluationSpec]

def run_standard_evaluation(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int,
    label: str
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Setup registry
    registry_file = root / "evaluation_registry.json"
    if registry_file.exists():
        logger.info(f"Reading {registry_file}")
        with registry_file.open("r") as fo:
            evaluation_registry = EvaluationRegistry.model_validate_json(fo.read())
        if label in evaluation_registry.evaluations:
            logger.info("Evaluation label already registered, skipping evaluation")
            return
    else:
        evaluation_registry = None
    evaluation_files = {}

    # Pair data
    logger.info(f"Running standard evaluation: {label}")
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
    logger.info("Scanning pairs")
    pairs = get_pairs_readers(startDT, endDT, root)
    start_string = startDT.strftime("%Y%m%d")
    end_string = endDT.strftime("%Y%m%d")
    for (d, c), data in pairs.items():
        # Check for domain
        if d not in evaluation_files:
            evaluation_files[d] = {}

        odir = root / f"parquet/{d}/evaluations"
        odir.mkdir(exist_ok=True)
        ofile = odir / f"{c}_{start_string}_{end_string}.parquet"

        # Add to registry
        evaluation_files[d][c] = ofile

        # Check for existence
        if ofile.exists():
            logger.info(f"Found {ofile}")
            continue

        # Run evaluation
        logger.info(f"Building {ofile}")
        if c in PREDICTION_RESAMPLING:
            # Handle forecasts
            # Group by feature id and lead time
            logger.info("Loading pairs")
            data = data.collect().to_pandas()

            logger.info("Grouping pairs")
            dataframes = [df for _, df in data.groupby(["nwm_feature_id", "lead_time_hours_min"])]
        else:
            # Handle simulations
            # Resolve duplicate predictions
            logger.info("Loading pairs")
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
                ).with_columns(pl.col("usgs_site_code").cast(pl.String)).collect().to_pandas()

            logger.info("Grouping pairs")
            dataframes = [df for _, df in data.groupby("nwm_feature_id")]

        # Evaluate
        logger.info("Computing metrics")
        chunk_size = max(1, len(dataframes) // jobs)
        results = pd.DataFrame.from_records(pool.map(bootstrap_metrics, dataframes, chunksize=chunk_size))
        
        # Save
        logger.info(f"Saving {ofile}")
        pl.DataFrame(results).write_parquet(ofile)

    # Clean-up
    logger.info("Cleaning up compute resources")
    pool.shutdown()

    # Register evaluation
    logger.info(f"Updating: {registry_file}")
    evaluation_spec = EvaluationSpec(
        label=label,
        startDT=startDT,
        endDT=endDT,
        directory=root,
        files=evaluation_files
    )
    if evaluation_registry is None:
        evaluation_registry = EvaluationRegistry(evaluations={label: evaluation_spec})
    else:
        evaluation_registry.evaluations[label] = evaluation_spec
    with registry_file.open("w") as fi:
        fi.write(evaluation_registry.model_dump_json(indent=4))
