"""Read-only methods."""
from pathlib import Path
import polars as pl
from nwm_explorer.mappings import Domain
from nwm_explorer.downloads import download_routelinks
from nwm_explorer.data import scan_routelinks

def read_routelinks(root: Path) -> dict[Domain, pl.LazyFrame]:
    """
    Lazily open routelink files as polars dataframes.
    
    Parameters
    ----------
    root: str | Path, optional, default "."
        Base data directory.
    
    Returns
    -------
    dict[Domain, pl.LazyFrame]
        Dataframes will only have columns found in schema. Keys are model
        Domain. Domain mapping uses schemas.DOMAIN_MAPPING by default.
        For example: ./parent/csv/RouteLink_HI.csv will be loaded and
        accessible using the key Domain.hawaii
    """
    return scan_routelinks(*download_routelinks(
        directory=root / "routelinks",
        read_only=True
    ))
