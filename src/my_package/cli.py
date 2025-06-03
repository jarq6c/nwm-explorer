import click
import pandas as pd
from my_package.my_module import make_dataframe
from my_package._version import __version__

@click.command()
@click.option("-c", "--ncols", "ncols", nargs=1, type=int, help="Number of columns", default=3)
@click.option("-r", "--nrows", "nrows", nargs=1, type=int, help="Number of rows", default=4)
def run(
    ncols: int,
    nrows: int
    ) -> None:
    """Generate a random dataframe and print to screen.

    Example:
    
    my-cli -c 3 -r 6
    """
    # Get a dataframe
    df = make_dataframe(ncols=ncols, nrows=nrows)

    # Print
    print(f"my_package {__version__}")
    with pd.option_context("display.precision", 2):
        print(df)

if __name__ == "__main__":
    run()
