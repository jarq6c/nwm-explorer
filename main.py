"""Generate static maps of evaluation metrics."""
from pathlib import Path

import polars as pl

from nwm_explorer.routelink import download_routelink
from nwm_explorer.evaluate import load_metrics
from nwm_explorer.constants import (DOMAIN_LOOKUP, CONFIGURATION_LOOKUP,
    METRIC_LOOKUP)
from nwm_explorer.views import build_lead_time_lookup

def main(
        root: Path,
        label: str = "FY2024-FY2025",
        rank: str = "max",
        threshold: str = "q85_cfs"
    ) -> None:
    """Main."""
    # Open routelink
    routelink = download_routelink(root).select(
        ["nwm_feature_id", "latitude", "longitude"]
    ).collect()

    # Merge domain and configuration look-ups
    configuration_lookup = {
        DOMAIN_LOOKUP[k]+v: (DOMAIN_LOOKUP[k], k) for k, v in CONFIGURATION_LOOKUP.items()
        }

    # Get lead time mapping
    lead_time_lookup = build_lead_time_lookup()

    # Iterate over model configurations
    for title, (domain, configuration) in configuration_lookup.items():
        # Iterate over lead times
        for lead_time in lead_time_lookup[configuration]:
            # Iterate over metrics
            for metric_label, metric in METRIC_LOOKUP.items():
                # Load data
                data = load_metrics(
                    root=root,
                    label=label,
                    configuration=configuration,
                    metric=metric,
                    lead_time_hours_min=lead_time,
                    rank=rank,
                    additional_columns=(
                        "nwm_feature_id",
                        "usgs_site_code",
                        "sample_size",
                        "observed_value_time_min",
                        "observed_value_time_max",
                        "reference_time_min",
                        "reference_time_max"
                        ),
                    threshold=threshold
                ).with_columns(
                    latitude=pl.col("nwm_feature_id").replace_strict(
                        old=routelink["nwm_feature_id"].implode(),
                        new=routelink["latitude"].implode()
                    ),
                    longitude=pl.col("nwm_feature_id").replace_strict(
                        old=routelink["nwm_feature_id"].implode(),
                        new=routelink["longitude"].implode()
                    )
                )

                print(data)
                return

if __name__ == "__main__":
    main(
        root=Path("/ised/nwm_explorer_data")
    )
