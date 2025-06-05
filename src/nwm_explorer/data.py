"""Methods to load and process data from various sources."""
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import xarray as xr
import numpy as np
import pandas as pd
import polars as pl

from nwm_explorer.mappings import (ROUTELINK_SCHEMA, DOMAIN_MAPPING, Domain,
    Configuration, FileType, Variable, Units, TIMEZONE_MAPPING)

def netcdf_validator(filepath: Path) -> None:
    """
    Validate that given filepath opens and closes without raising.

    Parameters
    ----------
    filepath: Path
        Path to file.
    
    Returns
    -------
    None
    """
    ds = xr.open_dataset(filepath)
    ds.close()

def csv_gz_validator(filepath: Path) -> None:
    """
    Validate that given filepath opens and closes without raising.

    Parameters
    ----------
    filepath: Path
        Path to file.
    
    Returns
    -------
    None
    """
    pd.read_csv(filepath, dtype=str)

def scan_routelinks(
        *filepaths: tuple[Path, ...],
        schema: dict[str, pl.DataType] = ROUTELINK_SCHEMA,
        domain_mapping: dict[str, Domain] = DOMAIN_MAPPING
        ) -> dict[Domain, pl.LazyFrame]:
    """
    Lazily open filepaths as polars dataframes.

    Parameters
    ----------
    filepaths: tuple[Path, ...], required
        One or more routelinks filepaths in CSV format to open.
    schema: dict[str, pl.DataType], optional
        Mapping from column frame labels to polars datatypes.
    domain_mapping: dict[str, Domain], optional
        Mapping from file names (Path.name) to Domain.
    
    Returns
    -------
    dict[Domain, pl.LazyFrame]
        Dataframes will only have columns found in schema. Keys are model
        Domain. Domain mapping uses schemas.DOMAIN_MAPPING by default.
        For example: ./parent/csv/RouteLink_HI.csv will be loaded and
        accessible using the key Domain.hawaii
    """
    frames = {}
    for fp in filepaths:
        domain = domain_mapping[fp.name]
        frames[domain] = pl.scan_csv(
            fp,
            comment_prefix="#",
            schema_overrides=schema
            ).select(list(schema.keys()))
        
        if domain == Domain.conus:
            frames[domain] = frames[domain].with_columns(
                    pl.col("usgs_site_code").replace("8313150", "08313150")
                )
    return frames

@dataclass
class NetCDFJob:
    """
    Input data for NetCDF processing jobs. Intended for use with National
    Water Model output.

    Attributes
    ----------
    filepaths: list[Path]
        List of filepaths to process.
    variables: list[str]
        Variables to extract from NetCDF Files.
    features: list[int]
        Feature to extract from NetCDF Files.
    """
    filepaths: list[Path]
    variables: list[str]
    features: list[int]

def process_netcdf(
        job: NetCDFJob
    ) -> pd.DataFrame:
    """
    Process a collection of National Water Model NetCDF files and return a
    dataframe.

    Parameters
    ----------
    job: NetCDFJob
        Job object used to track input files, target variables, and features.

    Returns
    -------
    pandas.DataFrame
    """
    with xr.open_mfdataset(job.filepaths) as ds:
        df = ds[job.variables].sel(feature_id=job.features
            ).to_dataframe().reset_index().dropna()
        if "time" not in df:
            df["time"] = ds.time.values[0]
        if "reference_time" not in df:
            df["reference_time"] = ds.reference_time.values[0]
        return df

def process_netcdf_parallel(
    filepaths: list[Path],
    variables: list[str],
    features: list[int],
    max_processes: int = 1,
    files_per_job: int = 5
    ) -> pd.DataFrame:
    """
    Process a collection of National Water Model NetCDF files and return a
    dataframe, in parallel.

    Parameters
    ----------
    filepaths: list[Path]
        List of filepaths to process.
    variables: list[str]
        Variables to extract from NetCDF Files.
    features: list[int]
        Feature to extract from NetCDF Files.
    max_processes: int, optional, default 1
        Maximum number of cores to use simultaneously.
    files_per_job: int, optional, default 5
        Maximum numer of files to load at once. Memory limited.

    Returns
    -------
    pandas.DataFrame
    """
    job_files = np.array_split(filepaths, len(filepaths) // files_per_job)
    jobs = [NetCDFJob(j, variables, features) for j in job_files]
    chunksize = max(1, len(jobs) // max_processes)
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        return pd.concat(pool.map(
            process_netcdf, jobs, chunksize=chunksize), ignore_index=True)

def process_nwis_tsv(filepath: Path) -> pd.DataFrame:
    """
    Process a NWIS IV API TSV file.

    Parameters
    ----------
    filepaths: list[Path]
        Path to file to process.

    Returns
    -------
    pandas.DataFrame
    """
    df = pd.read_csv(
        filepath,
        comment="#", 
        dtype=str,
        sep="\t",
        header=None,
        ).iloc[2:, 1:5]

    if df.iloc[:, -1].isna().all():
        return pd.DataFrame()

    df = df.set_axis(
        ["usgs_site_code", "value_time", "timezone", "value"],
        axis="columns")
    df = df[df["usgs_site_code"].str.isdigit()]
    df["value_time"] = pd.to_datetime(df["value_time"])
    
    # Deal with time zones
    for tz in df["timezone"].unique():
        mapped_tz = TIMEZONE_MAPPING.get(tz, tz)
        daylight = tz.endswith("DT")
        df.loc[df["timezone"] == tz, "value_time"] = df.loc[
            df["timezone"] == tz, "value_time"].dt.tz_localize(
                mapped_tz, ambiguous=daylight).dt.tz_convert(
                    "UTC").dt.tz_localize(None)

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["usgs_site_code", "value_time", "value"]].dropna()
    return df

def process_nwis_tsv_parallel(
        filepaths: list[Path],
        max_processes: int = 1
    ) -> pd.DataFrame:
    """
    Process a collection of USGS NWIS IV API TSV files and return a
    dataframe, in parallel.

    Parameters
    ----------
    filepaths: list[Path]
        List of filepaths to process.
    max_processes: int, optional, default 1
        Maximum number of cores to use simultaneously.

    Returns
    -------
    pandas.DataFrame
    """
    chunksize = max(1, len(filepaths) // max_processes)
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        df = pd.concat(pool.map(
            process_nwis_tsv, filepaths, chunksize=chunksize), ignore_index=True)
    df["usgs_site_code"] = df["usgs_site_code"].astype("category")
    return df

def generate_directory(
        root: Path,
        filetype: FileType,
        date_string: str,
        create: bool = True,
        configuration: Configuration | None = None
) -> Path:
    """
    Generate, and optionally create, a standardized directory path.
    
    Parameters
    ----------
    root: Path, required
        Root path.
    filetype: FileType, required
        Type of file.
    date_string: str, required
        Reference day folder.
    create: bool, optional, default True
        Create the directory.
    
    Returns
    -------
    Path
    """
    odir = root / filetype / date_string
    if configuration is not None:
        odir = odir / configuration

    if create:
        odir.mkdir(exist_ok=True, parents=True)
    return odir

def generate_filepath(
        root: Path,
        filetype: FileType,
        configuration: Configuration,
        variable: Variable,
        units: Units,
        date_string: str
) -> Path:
    """
    Generate a standardized file path.
    
    Parameters
    ----------
    root: Path, required
        Root path.
    filetype: FileType, required
        Type of file.
    domain: Domain, required
        Model domain.
    configuration: Configuration, required
        Model Configuration.
    variable: Variable, required
        Output variable.
    units: Units, required
        Measurement units.
    date_string: str, required
        Reference day folder.
    
    Returns
    -------
    Path
    """
    odir = generate_directory(root, filetype, date_string)
    filename = (
        f"{configuration}_"
        f"{variable}_{units}.{filetype}"
        )
    return odir / filename

def delete_directory(
        directory: Path,
        condition: Path
) -> None:
    """
    Delete a directory, conditionally.

    Parameters
    ----------
    directory: Path
        Directory to delete.
    condition: Path
        Path that must exist before deleting directory.
    
    Returns
    -------
    None
    """
    # Return if condition does not exist
    if not condition.exists():
        return
    
    # Return if directory does not exist
    if not directory.exists():
        return

    # Delete files
    files = list(directory.glob("**/*"))
    for f in files:
        f.unlink()
    directory.rmdir()
