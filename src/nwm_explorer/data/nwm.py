"""Testing interfaces and data models."""
from typing import Callable
from dataclasses import dataclass
from pathlib import Path
import inspect
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from nwm_explorer.data.download import download_files
from nwm_explorer.logging.logger import get_logger
from nwm_explorer.data.mapping import ModelDomain, ModelConfiguration

GOOGLE_CLOUD_BUCKET_URL: str = "https://storage.googleapis.com/national-water-model/"
"""National Water Model Google Cloud Storage bucket."""

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
    df = df.rename(columns={
        "time": "value_time",
        "feature_id": "nwm_feature_id",
        "streamflow": "predicted"
        })
    df["predicted"] = df["predicted"].astype(np.float32)
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
        base_url: str = GOOGLE_CLOUD_BUCKET_URL
) -> list[str]:
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

def analysis_assim_puertorico_no_da(
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

def medium_range_mem1(
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

def medium_range_alaska_mem1(
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

def short_range_puertorico(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_puertorico.
    """
    configuration = "short_range_puertorico/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(6, 30, 12)]
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

def short_range_puertorico_no_da(
        reference_date: pd.Timestamp
) -> list[str]:
    """
    Generate public urls for short_range_puertorico_no_da.
    """
    configuration = "short_range_puertorico_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(6, 30, 12)]
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

NWM_URL_BUILDERS: dict[tuple[ModelDomain, ModelConfiguration], Callable[[pd.Timestamp], list[str]]] = {
    (ModelDomain.alaska, ModelConfiguration.analysis_assim_extend_alaska_no_da): analysis_assim_extend_alaska_no_da,
    # (ModelDomain.conus, ModelConfiguration.analysis_assim_extend_no_da): analysis_assim_extend_no_da,
    # (ModelDomain.hawaii, ModelConfiguration.analysis_assim_hawaii_no_da): analysis_assim_hawaii_no_da,
    # (ModelDomain.puertorico, ModelConfiguration.analysis_assim_puertorico_no_da): analysis_assim_puertorico_no_da,
    # (ModelDomain.conus, ModelConfiguration.medium_range_mem1): medium_range_mem1,
    # (ModelDomain.conus, ModelConfiguration.medium_range_blend): medium_range_blend,
    # (ModelDomain.conus, ModelConfiguration.medium_range_no_da): medium_range_no_da,
    # (ModelDomain.alaska, ModelConfiguration.medium_range_alaska_mem1): medium_range_alaska_mem1,
    # (ModelDomain.alaska, ModelConfiguration.medium_range_blend_alaska): medium_range_blend_alaska,
    # (ModelDomain.alaska, ModelConfiguration.medium_range_alaska_no_da): medium_range_alaska_no_da,
    # (ModelDomain.conus, ModelConfiguration.short_range): short_range,
    # (ModelDomain.alaska, ModelConfiguration.short_range_alaska): short_range_alaska,
    # (ModelDomain.hawaii, ModelConfiguration.short_range_hawaii): short_range_hawaii,
    # (ModelDomain.hawaii, ModelConfiguration.short_range_hawaii_no_da): short_range_hawaii_no_da,
    # (ModelDomain.puertorico, ModelConfiguration.short_range_puertorico): short_range_puertorico,
    # (ModelDomain.puertorico, ModelConfiguration.short_range_puertorico_no_da): short_range_puertorico_no_da
}
"""Mapping from (ModelDomain, ModelConfiguration) to url builder function."""

def build_nwm_filepath(
    root: Path,
    domain: ModelDomain,
    configuration: ModelConfiguration,
    reference_date: pd.Timestamp
    ) -> Path:
    date_string = reference_date.strftime("nwm.%Y%m%d")
    return root / "parquet" / domain / date_string / f"{configuration}.parquet"

@dataclass
class PredictionFileDetails:
    domain: ModelDomain
    reference_date: pd.Timestamp
    configuration: ModelConfiguration
    path: Path
    builder: Callable[[pd.Timestamp], list[str]]
    features: np.ndarray[np.int64]

def build_nwm_file_details(
    root: Path,
    reference_dates: list[pd.Timestamp],
    features: dict[ModelDomain, np.ndarray[np.int64]]
    ) -> list[PredictionFileDetails]:
    """Returns list of file details matching parameters."""
    details = []
    for (d, c), b in NWM_URL_BUILDERS.items():
        for rd in reference_dates:
            details.append(
                PredictionFileDetails(
                    d, rd, c, build_nwm_filepath(root, d, c, rd), b, features[d]
                ))
    return details

def get_nwm_reader(root: Path, domain: ModelDomain) -> pl.LazyFrame:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)
    
    # Get file path
    fp = build_nwm_filepath(root, domain)
    logger.info(f"Scanning {fp}")
    return pl.scan_parquet(fp)

def get_nwm_readers(root: Path) -> dict[ModelDomain, pl.LazyFrame]:
    """Returns mapping from ModelDomain to polars.LazyFrame."""
    return 1
    return {d: get_nwm_reader(root, d) for d in nwm_FILENAMES}

def download_nwm(
    startDT: pd.Timestamp,
    endDT: pd.Timestamp,
    root: Path,
    routelinks: dict[ModelDomain, pl.LazyFrame],
    jobs: int
    ) -> None:
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Generate reference dates
    logger.info("Generating reference dates")
    reference_dates = generate_reference_dates(startDT, endDT)

    # Features to extract
    logger.info("Reading routelinks")
    features = {d: df.select("nwm_feature_id").collect()["nwm_feature_id"].to_numpy() for d, df in routelinks.items()}

    # File details
    logger.info("Generating file details")
    file_details = build_nwm_file_details(root, reference_dates, features)

    # Download
    logger.info("Preparing to download NWM data")
    temporary_directory = root / "temp"
    temporary_directory.mkdir(exist_ok=True)
    logger.info(f"Saving NetCDF files to {temporary_directory}")
    for fd in file_details:
        if fd.path.exists():
            logger.info(f"Skipping existing file {fd.path}")
            continue

        logger.info(f"Building {fd.path}")
        fd.path.parent.mkdir(exist_ok=True, parents=True)
        prefix = fd.reference_date.strftime("nwm.%Y%m%d_") + fd.domain + "_" + fd.configuration

        logger.info("Generating Google Cloud URLs")
        urls = fd.builder(fd.reference_date)
        file_paths = [temporary_directory / (prefix + f"_part_{i}.nc") for i in range(len(urls))]
        
        logger.info("Downloading NWM data")
        download_files(
            *list(zip(urls, file_paths)),
            timeout=3600,
            file_validator=netcdf_validator
        )

        logger.info("Processing NWM data")
        data = process_netcdf_parallel(
            file_paths,
            ["streamflow"],
            fd.features,
            jobs
        )
        logger.info(f"Saving {fd.path}")
        pl.DataFrame(data).write_parquet(fd.path)

        logger.info("Cleaning up NetCDF files")
        for fp in file_paths:
            if fp.exists():
                logger.info(str(fp))
                fp.unlink()

    # Clean-up
    logger.info(f"Cleaning up {temporary_directory}")
    temporary_directory.rmdir()
