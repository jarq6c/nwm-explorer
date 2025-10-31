"""
Download and process USGS streamflow observations.

Methods
-------
- download_site_table
- download_usgs
- scan_usgs
"""
from pathlib import Path
import inspect
import json
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from enum import StrEnum
from time import sleep
import functools

import us
import pandas as pd
import polars as pl
import geopandas as gpd

from .logger import get_logger
from .downloads import download_files
from .configuration import Configuration

LRU_CACHE_SIZE: int = 25
"""Maximum size of functools.lru_cache."""

NWIS_BASE_URL: str = (
    "https://waterservices.usgs.gov/nwis/iv/"
    "?format=json&siteStatus=all&parameterCd=00060"
)
"""NWIS IV API returning json and all site statuses."""

MONITORING_LOCATION_BASE_URL: str = (
    "https://api.waterdata.usgs.gov/ogcapi/v0/collections/monitoring-locations"
    "/items?f=json&lang=en-US&limit=10000&skipGeometry=false&offset=0"
    "&agency_code=USGS&site_type_code="
)
"""USGS monitoring location API returning geojson."""

class SiteTypeSlug(StrEnum):
    """Machine-friendly site types."""
    STREAM = "stream"
    CANAL = "canal"
    DITCH = "ditch"
    LAKE = "lake"
    TIDAL = "tidal"
    SPRING = "spring"
    DIVERSION = "diversion"
    ESTUARY = "estuary"
    TUNNEL = "tunnel"
    FIELD = "field"
    STORM_SEWER = "storm_sewer"
    COMBINED_SEWER = "combined_sewer"
    OUTFALL = "outfall"

@dataclass
class SiteType:
    """
    Dataclass for storing USGS site type details.

    Attributes
    ----------
    code: str
        Site type code.
    name: str
        Short name.
    long_name: str
        Long name.
    slug: SiteTypeSlug
        Machine-friendly representation.
    """
    code: str
    name: str
    long_name: str
    slug: SiteTypeSlug

SITE_TYPES: list[SiteType] = [
    SiteType("ST", "Stream", "Stream", SiteTypeSlug.STREAM),
    SiteType("ST-CA", "Canal", "Canal", SiteTypeSlug.CANAL),
    SiteType("ST-DCH", "Ditch", "Ditch", SiteTypeSlug.DITCH),
    SiteType("ST-TS", "Tidal SW", "Tidal stream", SiteTypeSlug.TIDAL),
    SiteType("LK", "Lake", "Lake, Reservoir, Impoundment", SiteTypeSlug.LAKE),
    SiteType("SP", "Spring", "Spring", SiteTypeSlug.SPRING),
    SiteType("FA-DV", "Diversion", "Diversion", SiteTypeSlug.DIVERSION),
    SiteType("ES", "Estuary", "Estuary", SiteTypeSlug.ESTUARY),
    SiteType("SB-TSM", "Tunl/mine", "Tunnel, shaft, or mine", SiteTypeSlug.TUNNEL),
    SiteType("FA-FON", "Agric area", "Field, Pasture, Orchard, or Nursery", SiteTypeSlug.FIELD),
    SiteType("FA-STS", "Sewer-strm", "Storm sewer", SiteTypeSlug.STORM_SEWER),
    SiteType("FA-CS", "Sewer-comb", "Combined sewer", SiteTypeSlug.COMBINED_SEWER),
    SiteType("FA-OF", "Outfall", "Outfall", SiteTypeSlug.OUTFALL)
]
"""List of USGS site types to retrieve for master site table."""

STATE_LIST: list[us.states.State] = us.states.STATES + [us.states.PR, us.states.DC]
"""List of US states."""

SUBDIRECTORY: str = "usgs"
"""Subdirectory that indicates root of USGS output parquet store."""

SITE_TABLE_DIRECTORY: str = "site_table"
"""Subdirectory that indicates root of USGS site parquet store."""

SITE_SCHEMA: pl.Schema = pl.Schema({
    "id": pl.String,
    "vertical_datum": pl.String,
    "original_horizontal_datum_name": pl.String,
    "well_constructed_depth": pl.String,
    "country_name": pl.String,
    "vertical_datum_name": pl.String,
    "drainage_area": pl.Float64,
    "hole_constructed_depth": pl.String,
    "minor_civil_division_code": pl.String,
    "hydrologic_unit_code": pl.String,
    "horizontal_positional_accuracy_code": pl.String,
    "contributing_drainage_area": pl.Float64,
    "depth_source_code": pl.String,
    "agency_name": pl.String,
    "basin_code": pl.String,
    "horizontal_positional_accuracy": pl.String,
    "time_zone_abbreviation": pl.String,
    "altitude": pl.Float64,
    "monitoring_location_name": pl.String,
    "district_code": pl.String,
    "state_code": pl.String,
    "site_type": pl.String,
    "horizontal_position_method_code": pl.String,
    "uses_daylight_savings": pl.String,
    "agency_code": pl.String,
    "country_code": pl.String,
    "county_code": pl.String,
    "altitude_accuracy": pl.Float64,
    "construction_date": pl.String,
    "aquifer_code": pl.String,
    "monitoring_location_number": pl.String,
    "state_name": pl.String,
    "site_type_code": pl.String,
    "altitude_method_code": pl.String,
    "horizontal_position_method_name": pl.String,
    "national_aquifer_code": pl.String,
    "county_name": pl.String,
    "altitude_method_name": pl.String,
    "original_horizontal_datum": pl.String,
    "aquifer_type_code": pl.String,
    "longitude": pl.Float64,
    "latitude": pl.Float64
})
"""Schema for USGS site data."""

def download_site_table(
        root: Path,
        config: Configuration
    ):
    """
    Download and process NWM output.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    config: Configuration
        Configuration object.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Output directory
    odir = root / SITE_TABLE_DIRECTORY
    odir.mkdir(exist_ok=True, parents=True)

    # Build site table
    logger.info("Building site table")
    for state in STATE_LIST:
        for site_type in SITE_TYPES:
            # Build state file
            ofile = odir / f"site_type_slug={site_type.slug}" / (state.abbr.lower() + ".parquet")
            if ofile.exists():
                logger.info("Found %s", ofile)
                continue
            logger.info("Buildling %s", ofile)
            ofile.parent.mkdir(exist_ok=True, parents=True)

            # Build URL
            url = (MONITORING_LOCATION_BASE_URL + site_type.code +
                f"&state_code={state.fips}" + f"&api_key={config.key}")
            logger.info("Downloading %s", url)

            # Fetch
            for attempt in range(3):
                try:
                    gdf = gpd.read_file(url)
                    break
                except Exception as e:
                    logger.info(e)
                    gdf = None
                    sleep(2 ** attempt)

            # Check for data
            if gdf is None:
                logger.info("Unable to retrieve")
                continue
            if gdf.empty:
                logger.info("Empty response")
                continue

            # Convert to polars
            gdf["longitude"] = gdf.geometry.x
            gdf["latitude"] = gdf.geometry.y
            data = pl.DataFrame(
                gdf.drop("geometry", axis=1),
                schema_overrides=SITE_SCHEMA
            )

            # Save
            logger.info("Saving %s", ofile)
            data.write_parquet(ofile)

def expand_site_table(
        ifile: Path,
        ofile: Path
) -> None:
    """
    Manually expand the site.

    Parameters
    ----------
    ifile: pathlib.Path
        GeoJSON file of USGS monitoring locations. (e.g.
        Path('new_york_sites.json'))
    ofile: pathlib.Path
        New parquet file to add to the application site table. (e.g.
        Path('site_table/site_type_slug=stream/ny_2.parquet'))
    
    Returns
    -------
    None
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Open file
    logger.info("Importing %s", ifile)
    gdf = gpd.read_file(ifile)

    # Convert to polars
    logger.info("Converting to parquet")
    gdf["longitude"] = gdf.geometry.x
    gdf["latitude"] = gdf.geometry.y
    data = pl.DataFrame(
        gdf.drop("geometry", axis=1),
        schema_overrides=SITE_SCHEMA
    )

    # Save
    logger.info("Saving %s", ofile)
    data.write_parquet(ofile)

def scan_site_table(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of USGS sites.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    """
    return pl.scan_parquet(
        root / SITE_TABLE_DIRECTORY,
        hive_schema={"site_type_slug": pl.Enum(SiteTypeSlug)}
        )

@functools.lru_cache(LRU_CACHE_SIZE)
def lookup_site_state_code_cache(root: Path, usgs_site_code: str) -> str:
    """
    Given a USGS site code, return the US state as a lower case two-character
    abbreviation.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    usgs_site_code: str
        USGS site code.
    
    Returns
    -------
    str
    """
    fips = scan_site_table(root).filter(
        pl.col("monitoring_location_number") == usgs_site_code
    ).select("state_code").collect()["state_code"].item(0)
    return us.states.lookup(fips).abbr.lower()

def lookup_site_state_code(root: Path, usgs_site_code: str) -> str:
    """
    Given a USGS site code, return the US state as a lower case two-character
    abbreviation.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    usgs_site_code: str
        USGS site code.
    
    Returns
    -------
    str
    """
    return lookup_site_state_code_cache(root, usgs_site_code)

def json_validator(ifile: Path) -> None:
    """
    Open and read json file data.
    """
    with ifile.open("r") as fi:
        json.loads(fi.read())

def build_usgs_filepath(
        root: Path,
        state_code: str,
        date: pd.Timestamp
) -> Path:
    """Build and return USGS hive-partitioned parquet file path."""
    # Build path
    state = f"state_code={state_code}"
    year = f"year={date.year}"
    month = f"month={date.month}"
    day = f"D{date.day}.parquet"
    return root / SUBDIRECTORY / state / year / month / day

def generate_usgs_url(
        date: pd.Timestamp,
        state_code: str
) -> str:
    """Returns download URL with parameters."""
    start = date.floor(freq="1d").strftime("%Y-%m-%dT%H:%MZ")
    end = (date.floor(freq="1d") + pd.Timedelta(hours=23, minutes=59)).strftime("%Y-%m-%dT%H:%MZ")
    return NWIS_BASE_URL + f"&stateCd={state_code}&startDT={start}&endDT={end}"

@dataclass
class JSONJob:
    """
    Parameters to keep track of input files and output files for processing
    USGS data.
    """
    ifile: Path
    ofile: Path

def process_json(job: JSONJob) -> None:
    """Process a JSON file."""
    # Load raw data
    with job.ifile.open("r") as fi:
        data = json.loads(fi.read())

    dfs = []
    for site in data["value"]["timeSeries"]:
        usgs_site_code = site["sourceInfo"]["siteCode"][0]["value"]
        for idx, series in enumerate(site["values"]):
            for value in series["value"]:
                # Update series
                value["usgs_site_code"] = usgs_site_code
                value["series"] = idx
                value["qualifiers"] = str(value["qualifiers"])
                dfs.append(value)
    pl.from_dicts(dfs).with_columns(
        pl.col("value").cast(pl.Float32),
        pl.col("usgs_site_code").cast(pl.String),
        pl.col("series").cast(pl.Int32),
        pl.col("qualifiers").cast(pl.String),
        pl.col("dateTime").str.to_datetime("%Y-%m-%dT%H:%M:%S%.3f%:z",
            time_unit="ms").dt.replace_time_zone(None)
    ).rename({
        "value": "observed_cfs",
        "dateTime": "value_time"
    }).drop_nulls("usgs_site_code").write_parquet(job.ofile)

def download_usgs(
        start: pd.Timestamp,
        end: pd.Timestamp,
        root: Path,
        retries: int = 3
    ):
    """
    Download and process NWM output.

    Parameters
    ----------
    start: pandas.Timestamp
        First reference date to retrieve and process.
    end: pandas.Timestamp
        Last refrence date to retrieve and process.
    routelink: polars.DataFrame
        Crosswalk from NWM channel feature IDs to USGS site codes.
    root: pathlib.Path
        Root data directory.
    retries: int, optional, default 3
        Number of times to retry NetCDF file downloads.
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # List of dates to retrieve
    reference_dates = pd.date_range(
        start=start.floor("1d"),
        end=end.ceil("1d"),
        freq="1d"
    )

    # Prepare downloads and output file paths
    urls: list[str] = []
    ofiles: list[Path] = []
    for state in STATE_LIST:
        # Generate file path and check for existence
        state_code = state.abbr.lower()

        # Process each reference day
        for rd in reference_dates:
            # Output file
            ofile = build_usgs_filepath(root, state_code, rd)

            # Skip existing files
            if ofile.exists():
                logger.info("Skipping existing file %s", ofile)
                continue

            logger.info("Preparing to build %s", ofile)
            ofiles.append(ofile)
            ofile.parent.mkdir(exist_ok=True, parents=True)
            urls.append(generate_usgs_url(rd, state_code))

    # Check for files
    if len(ofiles) == 0:
        logger.info("No files to download")
        return

    # Temporary directory
    odir = root / "temp"
    odir.mkdir(exist_ok=True)
    logger.info("Downloading to %s", odir)

    # Download and process
    with TemporaryDirectory(
        prefix="usgs_",
        dir=odir
        ) as td:
        for ofile, url in zip(ofiles, urls):
            # Download file
            json_file = Path(td) / str(ofile).replace(
                "/", "_").replace("=", "_").replace("parquet", "json")

            logger.info("Downloading %s", json_file)
            download_files(
                (url, json_file),
                limit=1,
                timeout=3600,
                headers={"Accept-Encoding": "gzip"},
                file_validator=json_validator,
                retries=retries
            )

            # Process
            logger.info("Building %s", ofile)
            process_json(JSONJob(json_file, ofile))

def scan_usgs_no_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    return pl.scan_parquet(
        root / f"{SUBDIRECTORY}/",
        hive_schema={
            "state_code": pl.Enum([s.abbr.lower() for s in STATE_LIST]),
            "year": pl.Int32,
            "month": pl.Int32
        }
    )

@functools.lru_cache(LRU_CACHE_SIZE)
def scan_usgs_cache(root: Path) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of USGS observations. Cache result.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.

    Returns
    -------
    polars.LazyFrame
    """
    return scan_usgs_no_cache(root)

def scan_usgs(root: Path, cache: bool = False) -> pl.LazyFrame:
    """
    Return polars.LazyFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    cache: bool, optional, default False
        Whether to cache the resulting LazyFrame for subsequent calls.

    Returns
    -------
    polars.LazyFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Scanning observations")
    if cache:
        return scan_usgs_cache(root)
    return scan_usgs_no_cache(root)

def load_usgs_no_cache(
    root: Path,
    state_code: str,
    year: int,
    month: int
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    state_code: str
        Two character lower case state abbreviation.
    year: int
        Integer year.
    month: int
        Integer month.
    
    Returns
    -------
    polars.DataFrame
    """
    return scan_usgs(root).filter(
        pl.col("state_code") == state_code,
        pl.col("year") == year,
        pl.col("month") == month
    ).select(
        ["usgs_site_code", "value_time", "observed_cfs"]
    ).collect().unique(
        ["usgs_site_code", "value_time"]
    ).sort(
        ["usgs_site_code", "value_time"]
    )

@functools.lru_cache(LRU_CACHE_SIZE)
def load_usgs_cache(
    root: Path,
    state_code: str,
    year: int,
    month: int
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations. Cache DataFrame.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    state_code: str
        Two character lower case state abbreviation.
    year: int
        Integer year.
    month: int
        Integer month.
    
    Returns
    -------
    polars.DataFrame
    """
    return scan_usgs(root, cache=True).filter(
        pl.col("state_code") == state_code,
        pl.col("year") == year,
        pl.col("month") == month
    ).select(
        ["usgs_site_code", "value_time", "observed_cfs"]
    ).collect().unique(
        ["usgs_site_code", "value_time"]
    ).sort(
        ["usgs_site_code", "value_time"]
    )

def load_usgs(
    root: Path,
    state_code: str,
    year: int,
    month: int,
    cache: bool = False
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    state_code: str
        Two character lower case state abbreviation.
    year: int
        Integer year.
    month: int
        Integer month.
    cache: bool, optional, default False
        Whether to cache the resulting DataFrame for subsequent calls.
    
    Returns
    -------
    polars.DataFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    logger.info("Retrieving observations")
    if cache:
        return load_usgs_cache(root, state_code, year, month)
    return load_usgs_no_cache(root, state_code, year, month)

def load_usgs_site(
    root: Path,
    usgs_site_code: str,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    cache: bool = False
    ) -> pl.DataFrame:
    """
    Return polars.DataFrame of USGS observations.

    Parameters
    ----------
    root: pathlib.Path
        Root data directory.
    usgs_site_code: str
        USGS site code.
    start_time: pandas.Timestamp
        Earliest datetime to retrieve.
    end_time: pandas.Timestamp
        Latest datetime to retrieve.
    cache: bool, optional, default False
        Whether to cache the underlying DataFrames for subsequent calls.
    
    Returns
    -------
    polars.DataFrame
    """
    # Get logger
    name = __loader__.name + "." + inspect.currentframe().f_code.co_name
    logger = get_logger(name)

    # Find state code
    logger.info("Looking up state code for %s", usgs_site_code)
    state_code = lookup_site_state_code(root, usgs_site_code)

    # Check end month
    if start_time.strftime("%Y%m") == end_time.strftime("%Y%m"):
        end_time += pd.Timedelta("31D")

    # Load data
    dataframes = []
    for m in pd.date_range(start_time, end_time, freq="1ME"):
        # Get month of data
        if cache:
            df = load_usgs_cache(root, state_code, m.year, m.month)
        else:
            df = load_usgs_no_cache(root, state_code, m.year, m.month)

        # Filter to relevant site, add to rest
        dataframes.append(df.filter(pl.col("usgs_site_code") == usgs_site_code))

    # Concatenate
    return pl.concat(dataframes).filter(
        pl.col("value_time") >= start_time,
        pl.col("value_time") <= end_time
    ).unique("value_time").sort("value_time")
