"""Command line interface utility."""
from sys import stderr
from pathlib import Path
import click
import pandas as pd
import polars as pl

from nwm_explorer._version import __version__
from nwm_explorer.data.routelink import download_routelinks, get_routelink_readers
from nwm_explorer.data.nwm import download_nwm, get_nwm_readers, get_nwm_reader, generate_reference_dates
from nwm_explorer.data.usgs import download_usgs, get_usgs_reader, get_usgs_readers
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.evaluation.compute import run_standard_evaluation, get_evaluation_reader

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
    "lead_time_hours_min": "Minimum lead time in hours"
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

@build_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-j", "--jobs", "jobs", nargs=1, required=False, type=click.INT, default=1, help="Maximum number of parallel processes (1)")
def build(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    directory: Path = Path("data"),
    jobs: int = 1
    ) -> None:
    """Retrieve and process required evaluation data."""
    # Download routelink, if missing
    download_routelinks(directory)

    # Scan routelinks
    routelinks = get_routelink_readers(directory)

    # Download NWM data, if needed
    download_nwm(startDT, endDT, directory, routelinks, jobs)

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
        routelinks,
        jobs
    )

@export_group.group()
def export():
    """Export predictions or observations to CSV."""
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
    
    nwm-explorer export predictions alaska analysis_assim_extend_alaska_no_da -s 2025-04-01 -e 2025-04-03 -o alaska_analysis_data.csv
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
    
    nwm-explorer export observations alaska -s 2025-04-01T12:00 -e 2025-04-02T02:15 -o alaska_usgs.csv
    """
    obs = get_usgs_reader(directory, domain, startDT, endDT)
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
    
    nwm-explorer export evaluations alaska analysis_assim_extend_alaska_no_da -s 2023-10-01 -e 2023-12-31 -o alaska_analysis_metrics.csv
    """
    results = get_evaluation_reader(domain, configuration, startDT, endDT, directory)
    try:
        write_to_csv(data=results, ofile=output, comments=comments, header=header)
    except FileNotFoundError:
        print(f"Data are unavailble for {domain} {configuration}", file=stderr)

@evaluation_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
@click.option("-j", "--jobs", "jobs", nargs=1, required=False, type=click.INT, default=1, help="Maximum number of parallel processes (1)")
def evaluate(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    directory: Path = Path("data"),
    jobs: int = 1
    ) -> None:
    """Run standard evaluations."""
    # Scan routelinks
    routelinks = get_routelink_readers(directory)

    # Download NWM data, if needed
    run_standard_evaluation(startDT, endDT, directory, routelinks, jobs)

cli = click.CommandCollection(sources=[
    build_group,
    export_group,
    evaluation_group
    ])

if __name__ == "__main__":
    cli()
