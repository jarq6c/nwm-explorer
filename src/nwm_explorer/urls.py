"""URLS to various data sources."""
from typing import Callable
import pandas as pd
from nwm_explorer.mappings import Domain, Configuration

ROUTELINKS_URL: str = (
    "https://www.hydroshare.org/resource/"
    "e9fe66730d184bdfbaea19639bd7cb55/data/"
    "contents/RouteLinks.tar.gz"
    )
"""NWM RouteLinks on HydroShare."""

GOOGLE_CLOUD_BUCKET_URL: str = "https://storage.googleapis.com/national-water-model/"
"""National Water Model Google Cloud Storage bucket."""

def generate_reference_dates(
        start: str | pd.Timestamp,
        end: str | pd.Timestamp
) -> list[str]:
    """
    Return list of formatted NWM GCS reference date subfolders from start
    date to end date.

    Parameters
    ----------
    start: str | Timestamp, required
        First date.
    end: str | Timestamp, required
        Last date
    
    Returns
    -------
    list[str]
        e.g. ['nwm.20250101/', 'nwm.20250102/']
    """
    return pd.date_range(
        start=start,
        end=end,
        freq="1d"
    ).strftime("nwm.%Y%m%d/").to_list()

def build_gcs_public_urls(
        reference_dates: list[str],
        configuration: str,
        prefixes: list[str],
        file_type: str,
        suffix: str,
        time_slices: list[str],
        base_url: str = GOOGLE_CLOUD_BUCKET_URL
) -> list[str]:
    urls = []
    for rd in reference_dates:
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
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_extend_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_hawaii_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def analysis_assim_puertorico_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_mem1(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_blend(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_alaska_mem1(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_blend_alaska(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def medium_range_alaska_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_alaska(
        reference_dates: list[str]
) -> list[str]:
    """
    Generate public urls for short_range_alaska.
    """
    configuration = "short_range_alaska/"
    prefixes_15 = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 6)]
    prefixes_45 = ["nwm.t" + str(p).zfill(2) + "z." for p in range(3, 27, 6)]
    file_type = "short_range.channel_rt."
    suffix = "conus.nc"
    time_slices_15 = ["f" + str(p).zfill(3) + "." for p in range(1, 16)]
    time_slices_45 = ["f" + str(p).zfill(3) + "." for p in range(1, 46)]
    urls_15 = build_gcs_public_urls(
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes_15,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices_15
    )
    urls_45 = build_gcs_public_urls(
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes_45,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices_45
    )
    return urls_15 + urls_45

def short_range_hawaii(
        reference_dates: list[str]
) -> list[str]:
    """
    Generate public urls for short_range_hawaii.
    """
    configuration = "short_range_hawaii/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 12)]
    file_type = "short_range.channel_rt."
    suffix = "hawaii.nc"
    time_slices = ["f" + str(t).zfill(5) + "." for t in range(15, 4815, 15)]
    return build_gcs_public_urls(
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_hawaii_no_da(
        reference_dates: list[str]
) -> list[str]:
    """
    Generate public urls for short_range_hawaii_no_da.
    """
    configuration = "short_range_hawaii_no_da/"
    prefixes = ["nwm.t" + str(p).zfill(2) + "z." for p in range(0, 24, 12)]
    file_type = "short_range_no_da.channel_rt."
    suffix = "hawaii.nc"
    time_slices = ["f" + str(t).zfill(5) + "." for t in range(15, 4815, 15)]
    return build_gcs_public_urls(
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_puertorico(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

def short_range_puertorico_no_da(
        reference_dates: list[str]
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
        reference_dates=reference_dates,
        configuration=configuration,
        prefixes=prefixes,
        file_type=file_type,
        suffix=suffix,
        time_slices=time_slices
    )

NWM_URL_BUILDERS: dict[tuple[Domain, Configuration], Callable] = {
    (Domain.alaska, Configuration.analysis_assim_extend_alaska_no_da): analysis_assim_extend_alaska_no_da,
    (Domain.conus, Configuration.analysis_assim_extend_no_da): analysis_assim_extend_no_da,
    (Domain.hawaii, Configuration.analysis_assim_hawaii_no_da): analysis_assim_hawaii_no_da,
    (Domain.puertorico, Configuration.analysis_assim_puertorico_no_da): analysis_assim_puertorico_no_da,
    (Domain.conus, Configuration.medium_range_mem1): medium_range_mem1,
    (Domain.conus, Configuration.medium_range_blend): medium_range_blend,
    (Domain.conus, Configuration.medium_range_no_da): medium_range_no_da,
    # (Domain.alaska, Configuration.medium_range_alaska_mem1): medium_range_alaska_mem1,
    # (Domain.alaska, Configuration.medium_range_blend_alaska): medium_range_blend_alaska,
    # (Domain.alaska, Configuration.medium_range_alaska_no_da): medium_range_alaska_no_da,
    # (Domain.conus, Configuration.short_range): short_range,
    # (Domain.alaska, Configuration.short_range_alaska): short_range_alaska,
    # (Domain.hawaii, Configuration.short_range_hawaii): short_range_hawaii,
    # (Domain.hawaii, Configuration.short_range_hawaii_no_da): short_range_hawaii_no_da,
    # (Domain.puertorico, Configuration.short_range_puertorico): short_range_puertorico,
    # (Domain.puertorico, Configuration.short_range_puertorico_no_da): short_range_puertorico_no_da
}
"""Mapping from (Domain, Configuration) to url builder function."""

def generate_usgs_urls(
        site_list: list[str],
        start_datetime: pd.Timestamp,
        end_datetime: pd.Timestamp
) -> list[str]:
    """
    Generate a list of USGS NWIS RDB URLs for each site in site_list.

    Parameters
    ----------
    site_list: list[str]
        List of USGS site codes.
    start_datetime: pd.Timestamp
        startDT.
    end_datetime: pd.Timestamp
        endDT.
    
    Returns
    -------
    list[str]
    """
    urls = []
    prefix = "https://waterservices.usgs.gov/nwis/iv/?format=rdb&"
    start_str = start_datetime.strftime("%Y-%m-%dT%H:%MZ")
    end_str = end_datetime.strftime("%Y-%m-%dT%H:%MZ")
    suffix = "&siteStatus=all&parameterCd=00060"

    for site in site_list:
        middle = f"sites={site}&startDT={start_str}&endDT={end_str}"
        urls.append(prefix+middle+suffix)
    return urls
