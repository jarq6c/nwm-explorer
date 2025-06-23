"""Read-only methods."""
from pathlib import Path
from dataclasses import dataclass
from typing import TypedDict, Any
import numpy as np
import pandas as pd
import polars as pl
import plotly.graph_objects as go

from nwm_explorer.mappings import (Domain, Configuration, Metric, Confidence,
    NWM_URL_BUILDERS, FileType, Variable, Units, EVALUATIONS, METRIC_SHORTHAND,
    CONFIDENCE_SHORTHAND, LEAD_TIME_FREQUENCY, DEFAULT_ZOOM)
from nwm_explorer.urls import generate_reference_dates
from nwm_explorer.data import generate_filepath, scan_routelinks
from nwm_explorer.logger import get_logger
from nwm_explorer.downloads import download_routelinks
from nwm_explorer.plotters import SiteMapPlotter

def read_NWM_output(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        low_memory: bool = False
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Find and lazily load NWM output.

    Parameters
    ----------
    root: Path, required
        Root directory storing downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    low_memory: bool, default False
        Reduce memory pressure at the expense of performance.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.readers.read_NWM_output")
    reference_dates = generate_reference_dates(start_date, end_date)

    # Read model output
    model_output = {}
    for (domain, configuration), _ in NWM_URL_BUILDERS.items():
        day_files = []
        for rd in reference_dates:
            # Check for file existence
            parquet_file = generate_filepath(
                root, FileType.parquet, configuration, Variable.streamflow,
                Units.cubic_feet_per_second, rd
            )
            logger.info(f"Building {parquet_file}")
            if parquet_file.exists():
                logger.info(f"Found existing {parquet_file}")
                day_files.append(parquet_file)
                continue
            
        # Check for at least one file
        if not day_files:
            logger.info(f"Found no data for {domain} {configuration}")
            continue
        
        # Merge files
        logger.info(f"Merging parquet files")
        model_output[(domain, configuration)] = pl.scan_parquet(day_files,
            low_memory=low_memory)
    return model_output

def read_USGS_observations(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[Domain, pl.LazyFrame]:
    """
    Scan and lazily load USGS observations.

    Parameters
    ----------
    root: Path, required
        Root directory storing downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[Domain, pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.pipelines.read_USGS_observations")
    # Read model output
    observations = {}
    s = start_date.strftime("%Y%m%dT%H")
    e = end_date.strftime("%Y%m%dT%H")
    for domain in list(Domain):
        # Check for file existence
        parquet_directory = root / FileType.parquet / Configuration.usgs
        parquet_file = parquet_directory / f"{domain}_{s}_{e}.parquet"
        logger.info(f"Building {parquet_file}")
        if parquet_file.exists():
            logger.info(f"Found existing {parquet_file}")
            observations[domain] = pl.scan_parquet(parquet_file)
    return observations

def scan_date_range(
        predictions: dict[tuple[Domain, Configuration], pl.LazyFrame]
) -> tuple[pd.Timestamp, pd.Timestamp]:
    first = None
    last = None
    for (domain, _), data in predictions.items():
        start = data.select("value_time").min().collect().item(0, 0)
        end = data.select("value_time").max().collect().item(0, 0)
        if first is None:
            first = start
            last = end
        else:
            first = min(first, start)
            last = max(last, end)
    return pd.Timestamp(first), pd.Timestamp(last)

def read_pairs(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Find and lazily load paired data.

    Parameters
    ----------
    root: Path, required
        Root directory storing downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.readers.read_pairs")
    # Process model output
    logger.info("Scanning pairs")
    reference_dates = generate_reference_dates(start_date, end_date)
    pairs = {}
    for (domain, configuration), _ in NWM_URL_BUILDERS.items():
        day_files = []
        for rd in reference_dates:
            # Check for file existence
            parquet_file = generate_filepath(
                root, FileType.parquet, configuration, Variable.streamflow_pairs,
                Units.cubic_feet_per_second, rd
            )
            if parquet_file.exists():
                logger.info(f"Found existing file: {parquet_file}")
                day_files.append(parquet_file)
                continue
            
        # Check for at least one file
        if not day_files:
            logger.info(f"Found no data for {domain} {configuration}")
            continue
        
        # Merge files
        logger.info(f"Merging parquet files")
        pairs[(domain, configuration)] = pl.scan_parquet(day_files)
    return pairs

def read_metrics(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Scan and lazily load metrics.

    Parameters
    ----------
    root: Path, required
        Root directory storing downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    logger = get_logger("nwm_explorer.pipelines.read_metrics")
    logger.info("Reading pairs")
    pairs = read_pairs(
        root=root,
        start_date=start_date,
        end_date=end_date
        )

    results = {}
    s = start_date.strftime("%Y%m%dT%H")
    e = end_date.strftime("%Y%m%dT%H")
    for (domain, configuration), paired in pairs.items():
        # Check for file existence
        parquet_directory = root / FileType.parquet / "metrics"
        parquet_file = parquet_directory / f"{domain}_{configuration}_{s}_{e}.parquet"
        logger.info(f"Building {parquet_file}")
        if parquet_file.exists():
            logger.info(f"Found existing {parquet_file}")
            results[(domain, configuration)] = pl.scan_parquet(parquet_file)
    return results

class FigurePatch(TypedDict):
    """
    A plotly figure patch.

    Keys
    ----
    data: list[go.Trace]
        A list of plotly traces.
    layout: go.Layout
        Plotly layout.
    """
    data: list[go.Trace]
    layout: go.Layout

@dataclass
class DashboardState:
    """Dashboard state variables."""
    evaluation: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    domain: Domain
    configuration: Configuration
    threshold: str
    metric: Metric
    metric_label: str
    confidence: Confidence
    lead_time: int

@dataclass
class MetricReader:
    """Intermediate metric reader to query and return data to dashboards."""
    root: Path
    last_domain: Domain | None = None

    def __post_init__(self) -> None:
        # Routelinks
        self.routelinks = scan_routelinks(*download_routelinks(
            self.root / "routelinks"))

        # Load metrics
        self.metrics: dict[str, dict[tuple[Domain, Configuration], pl.LazyFrame]] = {}
        for k, (s, e) in EVALUATIONS.items():
            self.metrics[k] = read_metrics(self.root, s, e)

        # Plot
        self.plotter = SiteMapPlotter()

    def query(self, state: DashboardState) -> pl.DataFrame | None:
        """Return data matching dashboard state."""
        try:
            data = self.metrics[state.evaluation][
                (state.domain, state.configuration)]
        except KeyError:
            return None
        crosswalk = self.routelinks[state.domain].select(
            ["usgs_site_code", "nwm_feature_id", "latitude", "longitude"]
        )
        col = METRIC_SHORTHAND[state.metric] + CONFIDENCE_SHORTHAND[state.confidence]
        if state.configuration in LEAD_TIME_FREQUENCY:
            data = data.filter(
                pl.col("lead_time_hours_min") == state.lead_time)

        data = data.select(
                [col, "nwm_feature_id"]
            ).join(
                crosswalk, on=["nwm_feature_id"], how="left"
            ).drop_nulls().rename({col: "value"})
        return data.collect()
    
    def get_plotly_patch(
            self,
            state: DashboardState,
            relayout_data: dict[str, Any] | None = None
            ) -> FigurePatch | None:
        """Return map of sites matching dashboard state."""
        data = self.query(state)
        if data is None:
            return data

        if state.domain == self.last_domain and relayout_data is not None:
            self.plotter.update_colors(
                values=data["value"].to_numpy(),
                label=state.metric_label,
                relayout_data=relayout_data
            )
        else:
            self.plotter.update_points(
                values=data["value"].to_numpy(),
                latitude=data["latitude"].to_numpy(),
                longitude=data["longitude"].to_numpy(),
                metric_label=state.metric_label,
                zoom=DEFAULT_ZOOM[state.domain],
                custom_data=data.select(["usgs_site_code", "nwm_feature_id"])
            )
            self.last_domain = state.domain
        return self.plotter.figure
