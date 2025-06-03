import click
import pandas as pd
from nwm_explorer._version import __version__
from nwm_explorer.mappings import Domain

class TimestampParamType(click.ParamType):
    name = "timestamp"

    def convert(self, value, param, ctx):
        if isinstance(value, pd.Timestamp):
            return value

        try:
            return pd.Timestamp(value)
        except ValueError:
            self.fail(f"{value!r} is not a valid timestamp", param, ctx)

analysis_group = click.Group()
obs_group = click.Group()

@analysis_group.command()
@click.argument("domain", nargs=1, required=True, type=click.Choice([d.value for d in Domain]))
# @click.option("-o", "--output", nargs=1, type=click.File("w"), help="Output file path", default="-")
# @click.option("-s", "--startDT", "startDT", nargs=1, type=TimestampParamType(), help="Start datetime")
# @click.option("-e", "--endDT", "endDT", nargs=1, type=TimestampParamType(), help="End datetime")
# @click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
# @click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
def analysis(
    domain: str, 
    # output: click.File,
    # startDT: pd.Timestamp = None,
    # endDT: pd.Timestamp = None,
    # parameterCd: str = "00060",
    # comments: bool = True,
    # header: bool = True
    ) -> None:
    """Retrieve NWM analysis data from Google Cloud and write in CSV format.

    Example:
    
    nwm-explorer analysis alaska
    """
    print(domain)

@analysis_group.command()
@click.argument("domain", nargs=1, required=True)
# @click.option("-o", "--output", nargs=1, type=click.File("w"), help="Output file path", default="-")
# @click.option("-s", "--startDT", "startDT", nargs=1, type=TimestampParamType(), help="Start datetime")
# @click.option("-e", "--endDT", "endDT", nargs=1, type=TimestampParamType(), help="End datetime")
# @click.option('--comments/--no-comments', default=True, help="Enable/disable comments in output, enabled by default")
# @click.option('--header/--no-header', default=True, help="Enable/disable header in output, enabled by default")
def obs(
    domain: str, 
    # output: click.File,
    # startDT: pd.Timestamp = None,
    # endDT: pd.Timestamp = None,
    # parameterCd: str = "00060",
    # comments: bool = True,
    # header: bool = True
    ) -> None:
    """Retrieve data from the USGS IV Web Service API and write in CSV format.

    Example:
    
    nwm-explorer obs alaska
    """
    print(domain)

cli = click.CommandCollection(sources=[
    analysis_group,
    obs_group
    ])

if __name__ == "__main__":
    cli()
