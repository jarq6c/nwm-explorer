from pathlib import Path
import click
import pandas as pd
import polars as pl
from nwm_explorer._version import __version__
from nwm_explorer.mappings import Domain, Configuration
from nwm_explorer.pipelines import (load_NWM_output, load_USGS_observations,
    load_metrics, load_pairs)
from nwm_explorer.downloads import download_routelinks
from nwm_explorer.data import scan_routelinks
from nwm_explorer.logger import get_logger
from nwm_explorer.gui import serve_dashboard
from nwm_explorer.readers import (read_NWM_output,
    read_USGS_observations, scan_date_range, read_metrics)

CSV_HEADERS: dict[str, str] = {
    "value_time": "Datetime of measurement or forecast valid time (UTC) (datetime string)",
    "variable": "Variable name (character string)",
    "usgs_site_code": "USGS Gage Site Code (character string)",
    "measurement_unit": "Units of measurement (character string)",
    "value": "Value quantity (float)",
    "qualifiers": "Qualifier string (character string)",
    "series": "Series number in case multiple time series are returned (integer)",
    "reference_time": "Forecast or analysis issue time or time zero (datetime string)",
    "nwm_feature_id": "NWM channel feature identifier (integer)",
    "nash_sutcliffe_efficiency": "Nash-Sutcliffe Model Efficiency Score (float)",
    "pearson_correlation_coefficient": "Pearson Linear Correlation Coefficient (float)",
    "mean_relative_bias": "Mean relative bias or mean relative error (float)",
    "relative_variability": "Ratio of predicted standard deviation to observed standard deviation (float)",
    "relative_mean": "Ratio of predicted mean to observed mean (float)",
    "sample_size": "Number of resampled pairs used to compute metrics (float)",
    "start_date": "Earliest valid time in evaluation pairs (float)",
    "end_date": "Latest valid time in evaluation pairs (float)",
    "kling_gupta_efficiency": "Kling-Gupta Model Efficiency Score (float)",
    "lead_time_days_min": "Minimum lead time in days. (integer)",
    "lead_time_hours_min": "Minimum lead time in hours. (integer)",
    "nse": "Nash-Sutcliffe model efficiency (float)",
    "nse_lower": "Nash-Sutcliffe model efficiency (Lower boundary of 95% confidence interval) (float)",
    "nse_upper": "Nash-Sutcliffe model efficiency (Upper boundary of 95% confidence interval) (float)",
    "rmb": "Relative mean bias (float)",
    "rmb_lower": "Relative mean bias (Lower boundary of 95% confidence interval) (float)",
    "rmb_upper": "Relative mean bias (Upper boundary of 95% confidence interval) (float)",
    "pearson": "Pearson correlation coefficient (float)",
    "pearson_lower": "Pearson correlation coefficient (Lower boundary of 95% confidence interval) (float)",
    "pearson_upper": "Pearson correlation coefficient (Upper boundary of 95% confidence interval) (float)",
    "rel_mean": "Relative mean (KGE component) (float)",
    "rel_mean_lower": "Relative mean (Lower boundary of 95% confidence interval) (float)",
    "rel_mean_upper": "Relative mean (Upper boundary of 95% confidence interval) (float)",
    "rel_var": "Relative variability (KGE component) (float)",
    "rel_var_lower": "Relative variability (Lower boundary of 95% confidence interval) (float)",
    "rel_var_upper": "Relative variability (Upper boundary of 95% confidence interval) (float)",
    "kge": "Kling-Gupta model efficiency (float)",
    "kge_lower": "Kling-Gupta model efficiency (Lower boundary of 95% confidence interval) (float)",
    "kge_upper": "Kling-Gupta model efficiency (Upper boundary of 95% confidence interval) (float)"
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

export_group = click.Group()
metrics_group = click.Group()
evaluate_group = click.Group()
display_group = click.Group()

@export_group.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice(Domain))
@click.argument("configuration", nargs=1, required=True, type=click.Choice(Configuration))
@click.option("-o", "--output", nargs=1, type=click.File("w", lazy=False), help="Output file path", default="-")
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
@click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def export(
    domain: Domain,
    configuration: Configuration,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data")
    ) -> None:
    """Export NWM evaluation data to CSV format.

    Example:
    
    nwm-explorer export alaska analysis_assim_extend_alaska_no_da -s 20231001 -e 20240101 -o alaska_analysis_data.csv
    """
    if configuration == Configuration.usgs:
        predictions = read_NWM_output(
            root=directory,
            start_date=startDT,
            end_date=endDT
        )
        first, last = scan_date_range(predictions)
        data = read_USGS_observations(
            root=directory,
            start_date=first,
            end_date=last
            )[domain]
    else:
        data = read_NWM_output(
            root=directory,
            start_date=startDT,
            end_date=endDT
        )[(domain, configuration)]
    
    # Write to CSV
    write_to_csv(data=data, ofile=output, comments=comments, header=header)

@metrics_group.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice(Domain))
@click.argument("configuration", nargs=1, required=True, type=click.Choice(list(Configuration)[:-1]))
@click.option("-o", "--output", nargs=1, type=click.File("w", lazy=False), help="Output file path", default="-")
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
@click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def metrics(
    domain: Domain,
    configuration: Configuration,
    output: click.File,
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    comments: bool = True,
    header: bool = True,
    directory: Path = Path("data")
    ) -> None:
    """Export NWM evaluation metrics to CSV format.

    Example:
    
    nwm-explorer metrics alaska analysis_assim_extend_alaska_no_da -s 20231001 -e 20240101 -o alaska_analysis_metrics.csv
    """
    data = read_metrics(
        root=directory,
        start_date=startDT,
        end_date=endDT
    )[(domain, configuration)]
    
    # Write to CSV
    write_to_csv(data=data, ofile=output, comments=comments, header=header,
        title="# NWM Explorer Metrics Export\n# \n")

@evaluate_group.command()
@click.option("-s", "--startDT", "startDT", nargs=1, required=True, type=TimestampParamType(), help="Start datetime")
@click.option("-e", "--endDT", "endDT", nargs=1, required=True, type=TimestampParamType(), help="End datetime")
@click.option("-d", "--directory", "directory", nargs=1, type=click.Path(path_type=Path), default="data", help="Data directory (./data)")
def evaluate(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    directory: Path = Path("data")
    ) -> None:
    """Run standard evaluation and generate parquet files.

    Example:
    
    nwm-explorer evaluate -s 20231001 -e 20240101
    """
    routelinks = scan_routelinks(*download_routelinks(directory / "routelinks"))
    predictions = load_NWM_output(
        root=directory,
        start_date=startDT,
        end_date=endDT,
        routelinks=routelinks
    )
    first, last = scan_date_range(predictions)
    load_USGS_observations(
        root=directory,
        start_date=first,
        end_date=last,
        routelinks=routelinks
    )
    load_pairs(
        root=directory,
        start_date=startDT,
        end_date=endDT
    )
    load_metrics(
        root=directory,
        start_date=startDT,
        end_date=endDT
    )

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
    export_group,
    metrics_group,
    evaluate_group,
    display_group
    ])

if __name__ == "__main__":
    cli()
