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
from nwm_explorer.data.nwm import NWM_URL_BUILDERS
from nwm_explorer.evaluation.metrics import bootstrap_metrics
from nwm_explorer.evaluation.pairs import get_pairs_readers, PREDICTION_RESAMPLING

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
    
    # Evaluate
    logger.info(f"Running standard evaluation: {label}")
    evaluation_files = {}

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
