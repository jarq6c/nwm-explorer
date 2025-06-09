"""Read-only methods."""
from pathlib import Path
from dataclasses import dataclass
import polars as pl
from nwm_explorer.mappings import Domain
from nwm_explorer.downloads import download_routelinks
from nwm_explorer.data import scan_routelinks

@dataclass
class RoutelinkReader:
    """
    Manages retrieval of crosswalk data from a collection of routelinks files.

    Attributes
    ----------
    root: Path
        Indicates root directory associated with nwm-explorer application.
    routelinks: dict[Domain, pl.LazyFrame]
        Dataframes will only have columns found in schema. Keys are model
        Domain. Domain mapping uses schemas.DOMAIN_MAPPING by default.
        For example: ./parent/csv/RouteLink_HI.csv will be loaded and
        accessible using the key Domain.hawaii
    """
    root: Path

    def __post_init__(self) -> None:
        """
        Lazily open routelink files as polars dataframes.
        """
        self.routelinks = scan_routelinks(*download_routelinks(
            directory=self.root / "routelinks",
            read_only=True
        ))
    
    @property
    def domains(self) -> list[Domain]:
        """List of available model domains."""
        return list(self.routelinks.keys())
    
    def site_list(self, domain: Domain) -> list[str]:
        """List of available USGS sites codes for given domain."""
        return self.routelinks[domain].select("usgs_site_code"
            ).collect()["usgs_site_code"].to_list()
    