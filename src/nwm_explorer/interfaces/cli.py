"""Command line interface utility."""
import os
from sys import stderr
from pathlib import Path
import click
import pandas as pd
import polars as pl

from nwm_explorer._version import __version__
from nwm_explorer.data.routelink import download_routelinks, get_routelink_readers
from nwm_explorer.data.nwm import download_nwm, get_nwm_readers, get_nwm_reader, generate_reference_dates
from nwm_explorer.data.usgs import download_usgs, get_usgs_reader
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import run_standard_evaluation, get_evaluation_reader
from nwm_explorer.interfaces.gui import serve_dashboard
from nwm_explorer.data.usgs_site_info import download_site_info
from nwm_explorer.evaluation.pairs import generate_pairs

CSV_HEADERS: dict[str, str] = {
    "value_time": "Valid time of observation or prediction (UTC).",
    "nwm_feature_id": "National Water Model channel feature ID. AKA reachID or comid.",
    "usgs_site_code": "USGS site code.",
    "reference_time": "Time of issuance for forecasts and model analyses (UTC).",
    "predicted": "Modeled streamflow, either forecast or analysis (ft^3/s).",
    "observed": "Gauge measured streamflow value (ft^3/s).",
    "nash_sutcliffe_efficiency_point": "Nash-Sutcliffe Model Effiency - single-valued point estimate",
    "nash_sutcliffe_efficiency_lower": "Nash-Sutcliffe Model Effiency - lower bound of 95% confidence interval",
    "nash_sutcliffe_efficiency_upper": "Nash-Sutcliffe Model Effiency - upper bound of 95% confidence interval",
    "relative_mean_bias_point": "Average ratio of prediction errors to observations - single-valued point estimate",
    "relative_mean_bias_lower": "Average ratio of prediction errors to observations - lower bound of 95% confidence interval",
    "relative_mean_bias_upper": "Average ratio of prediction errors to observations - upper bound of 95% confidence interval",
    "pearson_correlation_coefficient_point": "Linear correlation between predictions and observations - single-valued point estimate",
    "pearson_correlation_coefficient_lower": "Linear correlation between predictions and observations - lower bound of 95% confidence interval",
    "pearson_correlation_coefficient_upper": "Linear correlation between predictions and observations - upper bound of 95% confidence interval",
    "relative_mean_point": "Ratio of predicted mean to observed mean - single-valued point estimate",
    "relative_mean_lower": "Ratio of predicted mean to observed mean - lower bound of 95% confidence interval",
    "relative_mean_upper": "Ratio of predicted mean to observed mean - upper bound of 95% confidence interval",
    "relative_standard_deviation_point": "Ratio of predicted standard deviation to observed standard deviation - single-valued point estimate",
    "relative_standard_deviation_lower": "Ratio of predicted standard deviation to observed standard deviation - lower bound of 95% confidence interval",
    "relative_standard_deviation_upper": "Ratio of predicted standard deviation to observed standard deviation - upper bound of 95% confidence interval",
    "kling_gupta_efficiency_point": "Kling-Gupta Model Effiency - single-valued point estimate",
    "kling_gupta_efficiency_lower": "Kling-Gupta Model Effiency - lower bound of 95% confidence interval",
    "kling_gupta_efficiency_upper": "Kling-Gupta Model Effiency - upper bound of 95% confidence interval",
    "sample_size": "Number of samples used to compute metrics",
    "start_date": "Earliest valid time in sample pool",
    "end_date": "Latest valid time in sample pool",
    "lead_time_hours_min": "Minimum lead time in hours",
    "variable_name": "USGS variable name",
    "measurement_unit": "Units of measurement",
    "qualifiers": "USGS data quality codes",
    "series": "Identifying series number in the case of multiple time series (often due to multiple sensors)"
}
"""Column header descriptions."""

def write_to_csv(
    data: pl.LazyFrame,
    ofile: click.File,
    comments: bool = True,
    header: bool = True,
    title: str = "# NWM Explorer Data Export\n# \n"
    ) -> None:
    logger = get_logger("nwm_explorer.cli.write_to_csv")
    logger.info(f"Exporting to {ofile.name}")
    # Comments
    if comments:
        output = title
        
        for col in data.collect_schema().names():
            output += f"# {col}: {CSV_HEADERS.get(col, "UNKNOWN",)}\n"

        # Add version, link, and write time
        now = pd.Timestamp.utcnow()
        output += f"# \n# Generated at {now}\n"
        output += f"# nwm_explorer version: {__version__}\n"
        output += "# Source code: https://github.com/jarq6c/nwm_explorer\n# \n"

        # Write comments to file
        ofile.write(output)

    # Write data to file
    data.sink_csv(
        path=ofile,
        float_precision=2,
        include_header=header,
        batch_size=20000,
        datetime_format="%Y-%m-%dT%H:%M"
        )

class TimestampParamType(click.ParamType):
    name = "timestamp"

    def convert(self, value, param, ctx):
        if isinstance(value, pd.Timestamp):
            return value

        try:
            return pd.Timestamp(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid timestamp", param, ctx)

build_group = click.Group()
export_group = click.Group()
evaluation_group = click.Group()
display_group = click.Group()

@build_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-j", "--jobs", "jobs", nargs=1, required=False, type=click.INT, default=1, help="Maximum number of parallel processes (1)")
@click.option("-r", "--retries", "retries", nargs=1, required=False, type=click.INT, default=10, help="Maximum number of download retries (10)")
@click.option("-u", "--url", "url", nargs=1, type=click.STRING, default=None, help="National Water Model data server.")
def build(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    directory: Path = Path("data"),
    jobs: int = 1,
    retries: int = 10,
    url: str | None = None
    ) -> None:
    """Download and process data required by evaluations.

    Example:
    
    nwm-explorer build -s 2023-10-01 -e 2023-10-03 -j 4
    """
    # Download routelink, if missing
    download_routelinks(directory, retries=retries)

    # Scan routelinks
    routelinks = get_routelink_readers(directory)

    # Download supplemental site information
    download_site_info(directory, routelinks, jobs, retries)

    # Download NWM data, if needed
    if url is not None:
        os.environ["NWM_BASE_URL"] = url
    download_nwm(startDT, endDT, directory, routelinks, jobs, retries=retries)

    # Scan NWM data
    model_output = get_nwm_readers(startDT, endDT, directory)

    # Determine date range for observations
    first = startDT
    last = endDT
    for df in model_output.values():
        first = min(first, df.select("value_time").min().collect().item(0, 0))
        last = max(last, df.select("value_time").max().collect().item(0, 0))
    
    # Download observations, if needed
    download_usgs(
        pd.Timestamp(first),
        pd.Timestamp(last),
        directory,
        retries
    )

    # Pair data, if needed
    generate_pairs(
        startDT,
        endDT,
        directory,
        routelinks
    )

@export_group.group()
def export():
    """Export predictions, observations, or evaluations to CSV."""
    pass

@export.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice(ModelDomain))
@click.argument("configuration", nargs=1, required=True, type=click.Choice(ModelConfiguration))
@click.option("-o", "--output", nargs=1, type=click.File("w", lazy=False), help="Output file path", default="-")
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
@click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def predictions(
    domain: ModelDomain,
    configuration: ModelConfiguration,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data")
    ) -> None:
    """Export NWM evaluation data to CSV format.

    Example:
    
    nwm-explorer export predictions alaska analysis_assim_extend_alaska_no_da -s 2023-10-01 -e 2023-10-03 -o alaska_analysis_data.csv
    """
    reference_dates = generate_reference_dates(startDT, endDT)
    model_output = get_nwm_reader(directory, domain, configuration, reference_dates)
    try:
        write_to_csv(data=model_output, ofile=output, comments=comments, header=header)
    except FileNotFoundError:
        print(f"Data are unavailble for {domain} {configuration}", file=stderr)

@export.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice(ModelDomain))
@click.option("-o", "--output", nargs=1, type=click.File("w", lazy=False), help="Output file path", default="-")
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
@click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def observations(
    domain: ModelDomain,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data")
    ) -> None:
    """Export USGS observations to CSV format.

    Example:
    
    nwm-explorer export observations alaska -s 2023-10-01T12:00 -e 2023-10-03T02:15 -o alaska_usgs.csv
    """
    reference_dates = pd.date_range(
        start=startDT.floor("1d"),
        end=endDT.ceil("1d"),
        freq="1d"
    ).to_list()
    obs = get_usgs_reader(directory, domain, reference_dates).filter(
        pl.col("value_time") >= startDT,
        pl.col("value_time") <= endDT
    )
    try:
        write_to_csv(data=obs, ofile=output, comments=comments, header=header)
    except FileNotFoundError:
        print(f"Data are unavailble for {domain} usgs", file=stderr)

@export.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice(ModelDomain))
@click.argument("configuration", nargs=1, required=True, type=click.Choice(ModelConfiguration))
@click.option("-o", "--output", nargs=1, type=click.File("w", lazy=False), help="Output file path", default="-")
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
@click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def evaluations(
    domain: ModelDomain,
    configuration: ModelConfiguration,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data")
    ) -> None:
    """Export NWM evaluation metrics to CSV format.

    Example:
    
    nwm-explorer export evaluations alaska analysis_assim_extend_alaska_no_da -s 2023-10-01 -e 2023-10-03 -o alaska_analysis_eval.csv
    """
    results = get_evaluation_reader(domain, configuration, startDT, endDT, directory)
    try:
        write_to_csv(data=results, ofile=output, comments=comments, header=header)
    except FileNotFoundError:
        print(f"Data are unavailble for {domain} {configuration}", file=stderr)

@evaluation_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start date")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End date")
@click.option("-l", "--label", "label", nargs=1, type=click.STRING, default=None, help="Evaluation label")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-j", "--jobs", "jobs", nargs=1, required=False, type=click.INT, default=1, help="Maximum number of parallel processes (1)")
def evaluate(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    label: str | None = None,
    directory: Path = Path("data"),
    jobs: int = 1
    ) -> None:
    """Run a standard evaluation. Assumes build has already run and data are available.

    Example:
    
    nwm-explorer evaluate -s 2023-10-01 -e 2023-10-03 -j 4
    """
    # NOTE July 8th 12Z 2025, after 40 hour MRF mem1 was corrupted (incorrect reference time), add validation
    # Expand date range
    startDT = startDT.floor("1d")
    endDT = endDT.ceil("1d")

    # Set label
    if label is None:
        start_string = startDT.strftime("%Y%m%d")
        end_string = endDT.strftime("%Y%m%d")
        label = f"evaluation_{start_string}_{end_string}"

    # Download NWM data, if needed
    run_standard_evaluation(startDT, endDT, directory, jobs, label)

@display_group.command()
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-t", "--title", "title", nargs=1, type=click.STRING, default="National Water Model Evaluations", help="Dashboard title header")
def display(
    directory: Path = Path("data"),
    title: str = "National Water Model Evaluations"
    ) -> None:
    """Visualize and explore evaluation data.

    Example:
    
    nwm-explorer display
    """
    serve_dashboard(directory, title)

cli = click.CommandCollection(sources=[
    build_group,
    export_group,
    evaluation_group,
    display_group
    ])

if __name__ == "__main__":
    cli()
