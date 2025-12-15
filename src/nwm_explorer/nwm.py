"""
Download and process National Water Model output.

Methods
-------
- download_nwm
- scan_nwm
"""
from tempfile import TemporaryDirectory
from typing import Generator
from dataclasses import dataclass
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor
import os
import functools
import warnings

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from nwm_explorer.logger import get_logger
from nwm_explorer.downloads import download_files, FileValidationError
from nwm_explorer.routelink import download_routelink
from nwm_explorer.constants import (LRU_CACHE_SIZES, SUBDIRECTORIES, GOOGLE_CLOUD_BUCKET_URL,
    ModelConfiguration, ModelDomain, URLBuilder)

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

    Raises
    ------
    FileValidationError
    """
    try:
        with xr.open_dataset(filepath):
            return
    except Exception as e:
        raise FileValidationError(e) from e

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
    dfs = []
    for fp in job.filepaths:
        # Extract data
        with xr.open_dataset(fp) as ds:
            df = ds[job.variables].sel(feature_id=job.features
                ).to_dataframe().reset_index().dropna()
            if "time" not in df:
                df["time"] = ds.time.values[0]
            if "reference_time" not in df:
                df["reference_time"] = ds.reference_time.values[0]

        # Add to list
        dfs.append(df)

    # Merge
    merged = pd.concat(dfs, ignore_index=True)

    # Update columns
    merged = merged.rename(columns={
        "time": "value_time",
        "feature_id": "nwm_feature_id",
        "streamflow": "predicted_cfs"
        })

    # Downcast and convert to cubic feet per second
    merged["predicted_cfs"] = merged["predicted_cfs"].astype(np.float32) / (0.3048 ** 3.0)
    return merged

def process_netcdf_parallel(
    filepaths: list[Path],
    variables: list[str],
    features: list[int],
    max_processes: int = 1
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

    Returns
    -------
    pandas.DataFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    job_files = np.array_split(filepaths, max_processes)
    jobs = [NetCDFJob(j, variables, features) for j in job_files]
    with ProcessPoolExecutor(max_workers=max_processes) as pool:
        if len(jobs) == 0:
            logger.info("No jobs to process")
            return pd.DataFrame()

        try:
            return pd.concat(pool.map(process_netcdf, jobs), ignore_index=True)
        except ValueError as e:
            logger.info("Unable to process files %s", e)
    return pd.DataFrame()

def generate_reference_dates(
        start: str | pd.Timestamp,
        end: str | pd.Timestamp
) -> list[pd.Timestamp]:
    """
    Return list of pandas.Timestamp from start
    date to end date.

    Parameters
    ----------
    start: str | Timestamp, required
        First date.
    end: str | Timestamp, required
        Last date
    
    Returns
    -------
    list[pd.Timestamp]
    """
    return pd.date_range(
        start=start.tz_localize(None).floor(freq="1d"),
        end=end.tz_localize(None).floor(freq="1d"),
        freq="1d"
    ).to_list()

def build_gcs_public_urls(
        reference_date: pd.Timestamp,
        configuration: str,
        prefixes: list[str],
        file_type: str,
        suffix: str,
        time_slices: list[str],
        base_url: str | None = None
) -> list[str]:
    """Build and return a list of valid GCS public URLs."""
    if base_url is None:
        base_url = os.environ.get("NWM_BASE_URL", GOOGLE_CLOUD_BUCKET_URL)
    urls = []
    rd = reference_date.strftime("nwm.%Y%m%d/")
    for pf in prefixes:
        for ts in time_slices:
            urls.append(
                base_url +
                rd +
                configuration +
                pf +
                file_type +
                ts +
                suffix
                )
    return urls

def analysis_assim_extend_alaska_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for analysis_assim_extend_alaska_no_da.
    """
    configuration = "analysis_assim_extend_alaska_no_da/"
    prefixes = ["nwm.t20z."]
    file_type = "analysis_assim_extend_no_da.channel_rt."
    suffix = "alaska.nc"
    time_slices = ["tm" + str(t).zfill(2) + "." for t in range(8, 32)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_extend_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for analysis_assim_extend_no_da.
    """
    configuration = "analysis_assim_extend_no_da/"
    prefixes = ["nwm.t16z."]
    file_type = "analysis_assim_extend_no_da.channel_rt."
    suffix = "conus.nc"
    time_slices = ["tm" + str(t).zfill(2) + "." for t in range(4, 28)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for analysis_assim_no_da.
    """
    configuration = "analysis_assim_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24)]
    file_type = "analysis_assim_no_da.channel_rt."
    suffix = "conus.nc"
    time_slices = ["tm02."]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_hawaii_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for analysis_assim_hawaii_no_da.
    """
    configuration = "analysis_assim_hawaii_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24)]
    file_type = "analysis_assim_no_da.channel_rt."
    suffix = "hawaii.nc"
    time_slices = ["tm" + str(t).zfill(4) + "." for t in range(200, 260, 15)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_puerto_rico_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for analysis_assim_puertorico_no_da.
    """
    configuration = "analysis_assim_puertorico_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24)]
    file_type = "analysis_assim_no_da.channel_rt."
    suffix = "puertorico.nc"
    time_slices = ["tm02."]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_mem_1(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_mem1.
    """
    configuration = "medium_range_mem1/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range.channel_rt_1."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_blend_mem_1(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_blend_mem1.
    """
    configuration = "medium_range_blend_mem1/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_blend.channel_rt_1."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_ndfd(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_ndfd.
    """
    configuration = "medium_range_ndfd/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_ndfd.channel_rt."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_blend(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_blend.
    """
    configuration = "medium_range_blend/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_blend.channel_rt."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_no_da.
    """
    configuration = "medium_range_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_no_da.channel_rt."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(3, 241, 3)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_alaska_mem_1(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_alaska_mem1.
    """
    configuration = "medium_range_alaska_mem1/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range.channel_rt_1."
    suffix = "alaska.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_blend_alaska(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_blend_alaska.
    """
    configuration = "medium_range_blend_alaska/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_blend.channel_rt."
    suffix = "alaska.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 241)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_alaska_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for medium_range_alaska_no_da.
    """
    configuration = "medium_range_alaska_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "medium_range_no_da.channel_rt."
    suffix = "alaska.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(3, 241, 3)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range.
    """
    configuration = "short_range/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24)]
    file_type = "short_range.channel_rt."
    suffix = "conus.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 19)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_alaska(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_alaska.
    """
    configuration = "short_range_alaska/"
    prefixes_15 = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    prefixes_45 = ["nwm.t" + str(p).zfill(2) + "z." for p in range(3, 27, 6)]
    file_type = "short_range.channel_rt."
    suffix = "alaska.nc"
    time_slices_15 = ["f" + str(p).zfill(3) + "." for p in range(1, 16)]
    time_slices_45 = ["f" + str(p).zfill(3) + "." for p in range(1, 46)]
    urls_15 = build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes_15,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices_15
    )
    urls_45 = build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes_45,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices_45
    )
    return urls_15 + urls_45

def short_range_hawaii(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_hawaii.
    """
    configuration = "short_range_hawaii/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 12)]
    file_type = "short_range.channel_rt."
    suffix = "hawaii.nc"
    time_slices = []
    for h in range(0, 4900, 100):
        for m in range(0, 60, 15):
            time_slices.append("f" + str(h+m).zfill(5) + ".")
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices[1:-3]
    )

def short_range_hawaii_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_hawaii_no_da.
    """
    configuration = "short_range_hawaii_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 12)]
    file_type = "short_range_no_da.channel_rt."
    suffix = "hawaii.nc"
    time_slices = []
    for h in range(0, 4900, 100):
        for m in range(0, 60, 15):
            time_slices.append("f" + str(h+m).zfill(5) + ".")
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices[1:-3]
    )

def short_range_puerto_rico(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_puertorico.
    """
    configuration = "short_range_puertorico/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "short_range.channel_rt."
    suffix = "puertorico.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 49)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_puerto_rico_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_puertorico_no_da.
    """
    configuration = "short_range_puertorico_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    file_type = "short_range_no_da.channel_rt."
    suffix = "puertorico.nc"
    time_slices = ["f" + str(p).zfill(3) + "." for p in range(1, 49)]
    return build_gcs_public_urls(
        reference_date=reference_date,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

NWM_URL_BUILDERS: dict[tuple[ModelDomain, ModelConfiguration], URLBuilder] = {
    (ModelDomain.ALASKA,
     ModelConfiguration.ANALYSIS_ASSIM_EXTEND_ALASKA_NO_DA): analysis_assim_extend_alaska_no_da,
    (ModelDomain.CONUS,
     ModelConfiguration.ANALYSIS_ASSIM_EXTEND_NO_DA): analysis_assim_extend_no_da,
    (ModelDomain.CONUS,
     ModelConfiguration.ANALYSIS_ASSIM_NO_DA): analysis_assim_no_da,
    (ModelDomain.HAWAII,
     ModelConfiguration.ANALYSIS_ASSIM_HAWAII_NO_DA): analysis_assim_hawaii_no_da,
    (ModelDomain.PUERTO_RICO,
     ModelConfiguration.ANALYSIS_ASSIM_PUERTO_RICO_NO_DA): analysis_assim_puerto_rico_no_da,
    (ModelDomain.CONUS, ModelConfiguration.MEDIUM_RANGE_MEM_1): medium_range_mem_1,
    (ModelDomain.CONUS, ModelConfiguration.MEDIUM_RANGE_BLEND_MEM_1): medium_range_blend_mem_1,
    (ModelDomain.CONUS, ModelConfiguration.MEDIUM_RANGE_BLEND): medium_range_blend,
    (ModelDomain.CONUS, ModelConfiguration.MEDIUM_RANGE_NO_DA): medium_range_no_da,
    (ModelDomain.CONUS, ModelConfiguration.MEDIUM_RANGE_NDFD): medium_range_ndfd,
    (ModelDomain.ALASKA, ModelConfiguration.MEDIUM_RANGE_ALASKA_MEM_1): medium_range_alaska_mem_1,
    (ModelDomain.ALASKA, ModelConfiguration.MEDIUM_RANGE_BLEND_ALASKA): medium_range_blend_alaska,
    (ModelDomain.ALASKA, ModelConfiguration.MEDIUM_RANGE_ALASKA_NO_DA): medium_range_alaska_no_da,
    (ModelDomain.CONUS, ModelConfiguration.SHORT_RANGE): short_range,
    (ModelDomain.ALASKA, ModelConfiguration.SHORT_RANGE_ALASKA): short_range_alaska,
    (ModelDomain.HAWAII, ModelConfiguration.SHORT_RANGE_HAWAII): short_range_hawaii,
    (ModelDomain.HAWAII, ModelConfiguration.SHORT_RANGE_HAWAII_NO_DA): short_range_hawaii_no_da,
    (ModelDomain.PUERTO_RICO,
     ModelConfiguration.SHORT_RANGE_PUERTO_RICO): short_range_puerto_rico,
    (ModelDomain.PUERTO_RICO,
     ModelConfiguration.SHORT_RANGE_PUERTO_RICO_NO_DA): short_range_puerto_rico_no_da
}
"""Mapping from (ModelDomain, ModelConfiguration) to url builder function."""

def build_nwm_filepath(
    root: Path,
    configuration: ModelConfiguration,
    reference_date: pd.Timestamp
    ) -> Path:
    """Build and return NWM hive-partitioned parquet file path."""
    config = f"configuration={configuration}"
    year = f"year={reference_date.year}"
    month = f"month={reference_date.month}"
    day = f"D{reference_date.day}.parquet"
    subdirectory = SUBDIRECTORIES["nwm"]
    return root / subdirectory / config / year / month / day

def download_nwm(
        start: pd.Timestamp,
        end: pd.Timestamp,
        root: Path,
        jobs: int = 1,
        retries: int = 3,
        nwm_base_url: str | None = None,
        configuration: ModelConfiguration | None = None
) -> None:
    """
    Download and process NWM output.

    Parameters
    ----------
    start: pandas.Timestamp
        First reference date to retrieve and process.
    end: pandas.Timestamp
        Last refrence date to retrieve and process.
    root: pathlib.Path
        Root data directory.
    jobs: int, optional, default 1
        Number of parallel process used to process NWM output.
    retries: int, optional, default 3
        Number of times to retry NetCDF file downloads.
    nwm_base_url: str, optional
        Base URL from which to retrieve NWM NetCDF files.
    configuraton: ModelConfiguration, optional
        Limit download to a single model configuration.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Set base URL
    if nwm_base_url is not None:
        logger.info("NWM file server: %s", nwm_base_url)
        os.environ["NWM_BASE_URL"] = nwm_base_url

    # Load routelink
    routelink = download_routelink(root).select(
        ["nwm_feature_id", "domain"]
    ).collect()

    # Generate reference dates
    logger.info("Generating reference dates")
    reference_dates = generate_reference_dates(start, end)

    # Temporary directory
    odir = root / "temp"
    odir.mkdir(exist_ok=True)
    logger.info("Downloading to %s", odir)

    # Download and process data
    features = {}
    for (domain, config), builder in NWM_URL_BUILDERS.items():
        # Ignore other configurations
        if configuration and config != configuration:
            continue

        # Extract features
        if domain not in features:
            features[domain] = routelink.filter(
                pl.col("domain") == domain)["nwm_feature_id"].to_numpy()

        # Process each reference day
        for rd in reference_dates:
            # Output file
            ofile = build_nwm_filepath(root, config, rd)

            # Skip existing files
            if ofile.exists():
                logger.info("Skipping existing file %s", ofile)
                continue

            logger.info("Building %s", ofile)
            ofile.parent.mkdir(exist_ok=True, parents=True)

            logger.info("Generating NetCDF URLs")
            urls = builder(rd)

            # Temporary download directory
            with TemporaryDirectory(
                prefix=f"{config}_{rd.strftime("%Y%m%d")}_",
                dir=odir
                ) as td:
                file_paths = [Path(td) / f"part_{i}.nc" for i in range(len(urls))]

                logger.info("Downloading NWM data")
                download_files(
                    *list(zip(urls, file_paths)),
                    timeout=3600,
                    file_validator=netcdf_validator,
                    retries=retries,
                    limit=20
                )

                logger.info("Verifying files")
                verified = [f for f in file_paths if f.exists()]
                if not verified:
                    warnings.warn("No files verified", RuntimeWarning)
                    continue

                logger.info("Processing NWM data")
                data = process_netcdf_parallel(
                    verified,
                    ["streamflow"],
                    features[domain],
                    jobs
                )

                # Check for data
                if data.empty:
                    logger.info("No data to write")
                    continue

                logger.info("Saving %s", ofile)
                pl.DataFrame(data).with_columns(
                    pl.col("value_time").dt.cast_time_unit("ms"),
                    pl.col("reference_time").dt.cast_time_unit("ms")
                ).write_parquet(ofile)

def scan_nwm_no_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate reference dates
    logger.info("Scanning predictions")
    subdirectory = SUBDIRECTORIES["nwm"]
    return pl.scan_parquet(
        root / f"{subdirectory}/",
        hive_schema={
            "configuration": pl.Enum(ModelConfiguration),
            "year": pl.Int32,
            "month": pl.Int32
        }
    )

@functools.lru_cache(LRU_CACHE_SIZES["nwm"])
def scan_nwm_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of NWM output. Cache LazyFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    """
    return scan_nwm_no_cache(root)

def scan_nwm(root: Path, cache: bool = False) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    cache: bool, optional, default False
        If True, cache underlying LazyFrame.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Scanning predictions")
    if cache:
        return scan_nwm_cache(root)
    return scan_nwm_no_cache(root)

def load_nwm_no_cache(
    root: Path,
    configuration: ModelConfiguration,
    year: int,
    month: int,
    nwm_feature_id: int | None = None
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        Model configuration (e.g. ModelConfiguration.MEDIUM_RANGE_MEM_1)
    year: int
        Integer year.
    month: int
        Integer month.
    nwm_feature_id: int, optional
        NWM channel feature ID.
    
    Returns
    -------
    polars.DataFrame
    """
    # Initial scan
    df = scan_nwm(root).filter(
        pl.col("configuration") == configuration,
        pl.col("year") == year,
        pl.col("month") == month
    ).select(
        ["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    )

    # Feature by filter
    if nwm_feature_id is not None:
        df = df.filter(pl.col("nwm_feature_id") == nwm_feature_id)
    return df.collect().sort(
        ["nwm_feature_id", "reference_time", "value_time"]
    )

@functools.lru_cache(LRU_CACHE_SIZES["nwm"])
def load_nwm_cache(
    root: Path,
    configuration: ModelConfiguration,
    year: int,
    month: int,
    nwm_feature_id: int | None = None
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Cache result.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        Model configuration (e.g. ModelConfiguration.MEDIUM_RANGE_MEM_1)
    year: int
        Integer year.
    month: int
        Integer month.
    nwm_feature_id: int, optional
        NWM channel feature ID.
    
    Returns
    -------
    polars.DataFrame
    """
    # Initial scan with caching
    df = scan_nwm(root, cache=True).filter(
        pl.col("configuration") == configuration,
        pl.col("year") == year,
        pl.col("month") == month
    ).select(
        ["nwm_feature_id", "reference_time", "value_time", "predicted_cfs"]
    )

    # Feature by filter
    if nwm_feature_id is not None:
        df = df.filter(pl.col("nwm_feature_id") == nwm_feature_id)
    return df.collect().sort(
        ["nwm_feature_id", "reference_time", "value_time"]
    )

def load_nwm(
    root: Path,
    configuration: ModelConfiguration,
    year: int,
    month: int,
    cache: bool = False
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        Model configuration (e.g. ModelConfiguration.MEDIUM_RANGE_MEM_1)
    year: int
        Integer year.
    month: int
        Integer month.
    cache: bool, optional, default False
        If True, cache underlying dataframe.
    
    Returns
    -------
    polars.DataFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate reference dates
    logger.info("Retrieving predictions")
    if cache:
        return load_nwm_cache(root, configuration, year, month)
    return load_nwm_no_cache(root, configuration, year, month)

def nwm_site_generator(
    root: Path,
    configuration: ModelConfiguration,
    nwm_feature_id: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cache: bool = False
    ) -> Generator[pl.DataFrame]:
    """
    Iteratively return polars.DataFrame of NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        Model configuration (e.g. ModelConfiguration.MEDIUM_RANGE_MEM_1)
    nwm_feature_id: str
        NWM channel feature ID.
    start_time: pandas.Timestamp
        Earliest reference time to retrieve.
    end_time: pandas.Timestamp
        Latest reference time to retrieve.
    cache: bool, optional, default False
        Whether to cache the underlying DataFrames for subsequent calls.
    
    Returns
    -------
    polars.DataFrame
    """
    # Pad retrieval
    padded_start_time = start_time - pd.Timedelta("31D")
    padded_end_time = end_time + pd.Timedelta("31D")

    # Load data
    for m in pd.date_range(padded_start_time, padded_end_time, freq="1ME"):
        # Get month of data
        if cache:
            df = load_nwm_cache(root, configuration, m.year, m.month, nwm_feature_id)
        else:
            df = load_nwm_no_cache(root, configuration, m.year, m.month, nwm_feature_id)

        yield df.filter(
            pl.col("reference_time") >= start_time,
            pl.col("reference_time") <= end_time
        ).unique(["reference_time", "value_time"]).sort(["reference_time", "value_time"])

def load_nwm_site(
    root: Path,
    configuration: ModelConfiguration,
    nwm_feature_id: int,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cache: bool = False
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    configuration: ModelConfiguration
        Model configuration (e.g. ModelConfiguration.MEDIUM_RANGE_MEM_1)
    nwm_feature_id: str
        NWM channel feature ID.
    start_time: pandas.Timestamp
        Earliest reference time to retrieve.
    end_time: pandas.Timestamp
        Latest reference time to retrieve.
    cache: bool, optional, default False
        Whether to cache the underlying DataFrames for subsequent calls.
    
    Returns
    -------
    polars.DataFrame
    """
    # Collect dataframes
    dataframes = []
    for df in nwm_site_generator(
        root=root,
        configuration=configuration,
        nwm_feature_id=nwm_feature_id,
        start_time=start_time,
        end_time=end_time,
        cache=cache
    ):
        dataframes.append(df)
    return pl.concat(dataframes)
