"""Expected filenames after download and processing."""
from pathlib import Path

ROUTELINKS_MANIFEST: tuple[Path] = (
    Path("csv/RouteLink_HI.csv"),
    Path("csv/RouteLink_AK.csv"),
    Path("csv/RouteLink_PRVI.csv"),
    Path("csv/RouteLink_CONUS.csv")
)
"""Files expected to exist after extracting RouteLink tarball."""

def generate_default_manifest(
        n: int,
        directory: Path = Path("."),
        prefix: str = "part_",
        suffix: str = ".nc",
        zero_width: int = 5
        ) -> list[Path]:
    """
    Generates a list of generic paths.

    Parameters
    ----------
    n: int, required
        Number of filenames to generate.
    directory: Path, optional, default Path('.')
        Root directory.
    prefix: str, optional, default 'part_'
        Substring that appears at front of filename.
    suffix: str, optional, default '.nc'
        Substring that appears at back of filename, typically file extension.
    zero_width: int, optional, default 5
        Maximum width of zero-padded numerical string.

    Returns
    -------
    list[Path]
    """
    return [directory /
        (prefix + str(p).zfill(zero_width) + suffix) for p in range(n)]

def generate_usgs_manifest(
        site_list: list[str],
        directory: Path = Path(".")
        ) -> list[Path]:
    """
    Generates a list of paths to USGS TSVs.

    Parameters
    ----------
    site_list: list[str]
        List of USGS site codes.
    directory: Path, optional, default Path('.')
        Root directory.

    Returns
    -------
    list[Path]
    """
    return [directory / f"USGS_{s}.tsv.gz" for s in site_list]
