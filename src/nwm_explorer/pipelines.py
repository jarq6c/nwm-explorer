"""Various standard procedures."""
from pathlib import Path
import warnings
import pandas as pd
import polars as pl

from nwm_explorer.urls import generate_reference_dates, NWM_URL_BUILDERS, generate_usgs_urls
from nwm_explorer.manifests import generate_default_manifest, generate_usgs_manifest
from nwm_explorer.downloads import download_files, download_routelinks
from nwm_explorer.data import scan_routelinks, generate_filepath, generate_directory
from nwm_explorer.data import (process_netcdf_parallel, process_nwis_tsv_parallel,
    delete_directory)
from nwm_explorer.mappings import FileType, Variable, Units, Domain, Configuration
from nwm_explorer.metrics import (resample, nash_sutcliffe_efficiency,
    mean_relative_bias, pearson_correlation_coefficient, relative_mean,
    relative_variability, kling_gupta_efficiency)
from nwm_explorer.data import netcdf_validator, csv_gz_validator

def load_NWM_output(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        routelinks: dict[Domain, pl.LazyFrame]
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Download and process NWM output.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    routelinks: dict[Domain, LazyFrame]
        Mapping from Domain to crosswalk data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    reference_dates = generate_reference_dates(start_date, end_date)

    # Download and process model output
    model_output = {}
    for (domain, configuration), url_builder in NWM_URL_BUILDERS.items():
        # Check for file existence
        parquet_file = generate_filepath(
            root, FileType.PARQUET, domain, configuration, Variable.STREAMFLOW,
            Units.CUBIC_FEET_PER_SECOND, start_date, end_date
        )
        if parquet_file.exists():
            model_output[(domain, configuration)] = pl.scan_parquet(parquet_file)
            continue

        # Download
        urls = url_builder(reference_dates)
        download_directory = generate_directory(
            root, FileType.NETCDF, domain, configuration
        )
        manifest = generate_default_manifest(len(urls),
            directory=download_directory)
        download_files(*zip(urls, manifest), limit=10, timeout=3600,
            file_validator=netcdf_validator)
        
        # Validate manifest
        file_list = []
        for fp in manifest:
            if fp.exists():
                file_list.append(fp)
                continue
            warnings.warn(f"{fp} does not exist.", RuntimeWarning)

        # Process
        features = routelinks[domain].select(
            "nwm_feature_id").collect()["nwm_feature_id"].to_list()
        data = process_netcdf_parallel(
            filepaths=file_list,
            variables=["streamflow"],
            features=features,
            max_processes=12,
            files_per_job=25
        ).rename(columns={
                "time": "value_time",
                "feature_id": "nwm_feature_id",
                "streamflow": "value"
        })

        # Convert from cms to cfs
        data["value"] = data["value"].div(0.3048 ** 3.0)

        # Save to parquet
        pl.DataFrame(data).write_parquet(parquet_file)
        model_output[(domain, configuration)] = pl.scan_parquet(parquet_file)

        # Clean-up
        delete_directory(download_directory, parquet_file)
    return model_output

def load_USGS_observations(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        routelinks: dict[Domain, pl.LazyFrame]
) -> dict[Domain, pl.LazyFrame]:
    """
    Download and process USGS observations.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    routelinks: dict[Domain, LazyFrame]
        Mapping from Domain to crosswalk data.
    
    Returns
    -------
    dict[Domain, pl.LazyFrame]
    """
    # Download and process model output
    observations = {}
    for domain, rl in routelinks.items():
        # Check for file existence
        parquet_file = generate_filepath(
            root, FileType.PARQUET, domain, Configuration.OBSERVATIONS,
            Variable.STREAMFLOW, Units.CUBIC_FEET_PER_SECOND, start_date,
            end_date
        )
        if parquet_file.exists():
            observations[domain] = pl.scan_parquet(parquet_file)
            continue

        # Download
        sites = rl.select(
            "usgs_site_code").collect().to_pandas()["usgs_site_code"]
        sites = sites[sites.str.isdigit()].to_list()
        urls = generate_usgs_urls(
            sites, start_date, end_date
        )
        download_directory = generate_directory(
            root, FileType.TSV, domain, Configuration.OBSERVATIONS
        )
        manifest = generate_usgs_manifest(
            sites,
            directory=download_directory)
        download_files(*zip(urls, manifest), limit=10, timeout=3600, 
            headers={"Accept-Encoding": "gzip"}, auto_decompress=False,
            file_validator=csv_gz_validator)
        
        # Validate manifest
        file_list = []
        for fp in manifest:
            if fp.exists():
                file_list.append(fp)
                continue
            warnings.warn(f"{fp} does not exist.", RuntimeWarning)

        # Process
        data = process_nwis_tsv_parallel(
            filepaths=file_list,
            max_processes=12
        ).rename(columns={
                "time": "value_time",
                "feature_id": "nwm_feature_id",
                "streamflow": "value"
        })

        # Save to parquet
        pl.DataFrame(data).write_parquet(parquet_file)
        observations[domain] = pl.scan_parquet(parquet_file)

        # Clean-up
        delete_directory(download_directory, parquet_file)
    return observations

def load_pairs(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Download and process NWM and USGS output.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    routelinks = scan_routelinks(*download_routelinks(root / "routelinks"))

    observations = load_USGS_observations(
        root=root,
        start_date=start_date,
        end_date=end_date,
        routelinks=routelinks
    )
    predictions = load_NWM_output(
        root=root,
        start_date=start_date,
        end_date=end_date,
        routelinks=routelinks
    )

    pairs = {}
    for (domain, configuration), data in predictions.items():
        # Check for file existence
        parquet_file = generate_filepath(
            root, FileType.PARQUET, domain, configuration, Variable.STREAMFLOW_PAIRS,
            Units.CUBIC_FEET_PER_SECOND, start_date, end_date
        )
        if parquet_file.exists():
            pairs[(domain, configuration)] = pl.scan_parquet(parquet_file)
            continue

        # Pair data
        crosswalk = routelinks[domain].select(["nwm_feature_id",
            "usgs_site_code"]).collect()
        obs = observations[domain].with_columns(
            pl.col("usgs_site_code").cast(pl.String)
        )
        paired_data = data.with_columns(
            usgs_site_code=pl.col("nwm_feature_id").replace_strict(
                crosswalk["nwm_feature_id"], crosswalk["usgs_site_code"])
        ).join(obs, on=["usgs_site_code", "value_time"], how="left",
                suffix="_obs").drop_nulls().rename({"value": "value_pred"})

        # Save to parquet
        paired_data.collect().write_parquet(parquet_file)
        pairs[(domain, configuration)] = pl.scan_parquet(parquet_file)
    return pairs

def load_metrics(
        root: Path,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
) -> dict[tuple[Domain, Configuration], pl.LazyFrame]:
    """
    Download and process NWM and USGS output.

    Parameters
    ----------
    root: Path, required
        Root directory to save downloaded and processed files.
    start_date: pd.Timestamp
        First date to start retrieving data.
    end_date: pd.Timestamp
        Last date to retrieve data.
    
    Returns
    -------
    dict[tuple[Domain, Configuration], pl.LazyFrame]
    """
    pairs = load_pairs(
        root=root,
        start_date=start_date,
        end_date=end_date
        )

    results = {}
    for (domain, configuration), data in pairs.items():
        # Check for file existence
        parquet_file = generate_filepath(
            root, FileType.PARQUET, domain, configuration, Variable.STREAMFLOW_METRICS,
            Units.METRICS, start_date, end_date
        )
        if parquet_file.exists():
            results[(domain, configuration)] = pl.scan_parquet(parquet_file)
            continue

        daily_max = resample(data)
        metric_results = daily_max.group_by(
            "usgs_site_code").agg(
            pl.struct(["value_obs", "value_pred"])
            .map_batches(
                lambda combined: nash_sutcliffe_efficiency(
                    combined.struct.field("value_obs"),
                    combined.struct.field("value_pred")
                ),
                returns_scalar=True
            )
            .alias("NSE"),
            pl.struct(["value_obs", "value_pred"])
            .map_batches(
                lambda combined: pearson_correlation_coefficient(
                    combined.struct.field("value_obs"),
                    combined.struct.field("value_pred")
                ),
                returns_scalar=True
            )
            .alias("PCC"),
            pl.struct(["value_obs", "value_pred"])
            .map_batches(
                lambda combined: mean_relative_bias(
                    combined.struct.field("value_obs"),
                    combined.struct.field("value_pred")
                ),
                returns_scalar=True
            )
            .alias("MRB"),
            pl.struct(["value_obs", "value_pred"])
            .map_batches(
                lambda combined: relative_variability(
                    combined.struct.field("value_obs"),
                    combined.struct.field("value_pred")
                ),
                returns_scalar=True
            )
            .alias("relative_variability"),
            pl.struct(["value_obs", "value_pred"])
            .map_batches(
                lambda combined: relative_mean(
                    combined.struct.field("value_obs"),
                    combined.struct.field("value_pred")
                ),
                returns_scalar=True
            )
            .alias("relative_mean"),
            pl.col("value_obs").count().alias("sample_size"),
            pl.col("value_time").min().alias("start_date"),
            pl.col("value_time").max().alias("end_date")
        ).with_columns(
            kling_gupta_efficiency()
        )

        # Save to parquet
        metric_results.collect().write_parquet(parquet_file)
        results[(domain, configuration)] = pl.scan_parquet(parquet_file)
    return results
