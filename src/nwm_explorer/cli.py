"""Command-line Interface."""
from pathlib import Path
from datetime import datetime
from typing import Literal
from sys import stdout

import typer
import pandas as pd
import polars as pl

from nwm_explorer.configuration import load_configuration
from nwm_explorer.routelink import download_routelink
from nwm_explorer.nwm import download_nwm
from nwm_explorer.usgs import download_usgs
from nwm_explorer.pairs import pair_nwm_usgs
from nwm_explorer.evaluate import evaluate as run_evaluation
from nwm_explorer.evaluate import scan_evaluations
from nwm_explorer.constants import (ModelConfiguration, EvaluationMetric,
    COLUMN_DESCRIPTIONS, NO_THRESHOLD_LABEL, METRIC_PLOTTING_LIMITS)
from nwm_explorer.gui import serve_dashboards

app = typer.Typer()
"""Main typer command-line application."""

@app.command()
def download(
    start: datetime,
    end: datetime,
    root: Path,
    jobs: int = 1,
    retries: int = 3,
    configuration: ModelConfiguration | None = None,
    observations: bool = True,
    nwm_base_url: str | None = None
    ) -> None:
    """
    Download and process NWM and USGS data for evaluations.
    """
    # Use time stamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Make root
    root.mkdir(exist_ok=True, parents=True)

    # Routelink
    download_routelink(root=root)

    # NWM downloads
    download_nwm(
        start=start,
        end=end,
        root=root,
        jobs=jobs,
        retries=retries,
        nwm_base_url=nwm_base_url,
        configuration=configuration
    )

    # USGS downloads
    if observations:
        download_usgs(
            start=start-pd.Timedelta("1d"),
            end=end+pd.Timedelta("10d"),
            root=root,
            retries=retries
        )

@app.command()
def pair(
    start: datetime,
    end: datetime,
    root: Path
    ) -> None:
    """
    Resample and pair NWM predictions to USGS observations.
    """
    # Use timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Pair
    pair_nwm_usgs(
        root=root,
        start_date=start,
        end_date=end
    )

@app.command()
def compute(
    start: datetime,
    end: datetime,
    root: Path,
    label: str,
    jobs: int = 1,
    sites_per_chunk: int = 500,
    threshold_file: Path | None = None,
    threshold_column: list[str] | None = None
    ) -> None:
    """
    Compute evaluation metrics for NWM-USGS pairs.
    """
    # Use timestamps
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)

    # Evaluate
    run_evaluation(
        label=label,
        root=root,
        start_time=start,
        end_time=end,
        processes=jobs,
        sites_per_chunk=sites_per_chunk,
        threshold_file=threshold_file,
        threshold_columns=threshold_column
    )

@app.command()
def evaluate(configuration: Path = Path("config.json")) -> None:
    """
    Run standard evaluation including download, pair, and compute. Parameters
    set in configuration file.
    """
    # Load configuration
    config = load_configuration(configuration_file=configuration)

    # Make root
    config.root.mkdir(exist_ok=True, parents=True)

    # Routelink
    download_routelink(root=config.root)

    # Process each evaluation
    for e in config.evaluations:
        # Set start and end times
        start = pd.Timestamp(e.start_time)
        end = pd.Timestamp(e.end_time)

        # NWM downloads
        download_nwm(
            start=start,
            end=end,
            root=config.root,
            jobs=config.processes,
            retries=config.retries,
            nwm_base_url=config.nwm_base_url,
            skip=e.skip
        )

        # USGS downloads
        download_usgs(
            start=start-pd.Timedelta("1d"),
            end=end+pd.Timedelta("10d"),
            root=config.root,
            retries=config.retries
        )

        # Pair
        pair_nwm_usgs(
            root=config.root,
            start_date=start,
            end_date=end
        )

        # Evaluate
        run_evaluation(
            label=e.label,
            root=config.root,
            start_time=start,
            end_time=end,
            processes=config.processes,
            sites_per_chunk=config.sites_per_chunk,
            threshold_file=config.threshold_file,
            threshold_columns=config.threshold_columns
        )

@app.command()
def export(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: EvaluationMetric,
        threshold: str = NO_THRESHOLD_LABEL,
        output: typer.FileTextWrite | None = None,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median",
        additional_columns: tuple[str] | None = None,
        no_data_value: str | None = None
    ) -> None:
    """
    Export evaluation metrics to CSV.
    """
    # Set output
    if output is None:
        output = stdout

    # Set additional columns
    if additional_columns is None:
        additional_columns = [
            "label",
            "configuration",
            "nwm_feature_id",
            "usgs_site_code",
            "reference_time_min",
            "reference_time_max",
            "lead_time_hours_min",
            "predicted_value_time_min",
            "predicted_value_time_max",
            f"predicted_cfs_{rank}",
            "observed_value_time_min",
            "observed_value_time_max",
            f"observed_cfs_{rank}",
            "sample_size"
        ]
        if threshold != NO_THRESHOLD_LABEL:
            additional_columns += ["threshold", "threshold_value"]

    # Load metrics
    data = scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min,
        pl.col("threshold") == threshold
    ).select(
        additional_columns + [
            f"{metric}_{rank}_lower",
            f"{metric}_{rank}_point",
            f"{metric}_{rank}_upper"
        ]
    ).collect()

    # Write header
    header = "# National Water Model Evaluations\n# \n"
    for col in data.columns:
        header += f"# {col}: {COLUMN_DESCRIPTIONS.get(col)}\n"
    header += "# \n"
    output.write(header)

    # Write data
    output.write(data.write_csv(
        float_precision=2,
        datetime_format="%Y-%m-%dT%H:%M",
        null_value=no_data_value
    ))

@app.command()
def display(configuration: Path = Path("config.json")) -> None:
    """
    Launch graphical application in the browser. Shutdown the application using
    ctrl+c.
    """
    serve_dashboards(
        configuration_file=configuration
    )

@app.command()
def plot(
        root: Path,
        label: str,
        configuration: ModelConfiguration,
        metric: EvaluationMetric,
        threshold: str = NO_THRESHOLD_LABEL,
        lead_time_hours_min: int = 0,
        rank: Literal["min", "median", "max"] = "median"
    ) -> None:
    """
    Plot evaluation results on a map.
    """
    # Look-up plotting limits
    cmin, cmax = METRIC_PLOTTING_LIMITS[str(metric)]

    # Load metrics, drop or fill missing values
    data = scan_evaluations(
        root
    ).filter(
        pl.col("label") == label,
        pl.col("configuration") == configuration,
        pl.col("lead_time_hours_min") == lead_time_hours_min,
        pl.col("threshold") == threshold
    ).select(
        f"{metric}_{rank}_lower",
        f"{metric}_{rank}_point",
        f"{metric}_{rank}_upper"
    ).collect().drop_nulls(
        subset=f"{metric}_{rank}_point"
    ).with_columns(
        pl.col(f"{metric}_{rank}_lower").fill_null(cmin),
        pl.col(f"{metric}_{rank}_upper").fill_null(cmax)
    ).with_columns(
        pl.when(
            pl.col(f"{metric}_{rank}_lower") > pl.col(f"{metric}_{rank}_point")
            ).then(
                pl.col(f"{metric}_{rank}_point")
                ).otherwise(pl.col(f"{metric}_{rank}_lower")
        ).alias(f"{metric}_{rank}_lower")
    )

    # Plot metrics
    print(data)

def run() -> None:
    """Main entry point for CLI."""
    app()

if __name__ == "__main__":
    # Run the CLI.
    run()
