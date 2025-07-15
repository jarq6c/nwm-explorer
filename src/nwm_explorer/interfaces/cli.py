"""Command line interface utility."""
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

CSV_HEADERS: dict[str, str] = {
    "value_time": "Valid time of observation or prediction (UTC).",
    "nwm_feature_id": "National Water Model channel feature ID. AKA reachID or comid.",
    "usgs_site_code": "USGS site code.",
    "reference_time": "Time of issuance for forecasts and model analyses (UTC).",
    "predicted": "Modeled streamflow, either forecast or analysis (ft^3/s).",
    "observed": "Gauge measured streamflow value (ft^3/s)."
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
            output += f"# {col}: {CSV_HEADERS.get(col, "UNKNOWN")}\n"

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
export_group = click.Group("export")

@build_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data-new", help="Data directory (./data-new)")
@click.option("-j", "--jobs", "jobs", nargs=1, required=False, type=click.INT, default=1, help="Maximum number of parallel processes (1)")
def build(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    directory: Path = Path("data-new"),
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
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data-new", help="Data directory (./data-new)")
def predictions(
    domain: ModelDomain,
    configuration: ModelConfiguration,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data-new")
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
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data-new", help="Data directory (./data-new)")
def observations(
    domain: ModelDomain,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data-new")
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

cli = click.CommandCollection(sources=[
    build_group,
    export_group
    ])

if __name__ == "__main__":
    cli()
