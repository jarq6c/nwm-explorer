"""Command-line Interface."""
from pathlib import Path

import pandas as pd

from modules.evaluate import evaluate
from modules.usgs import download_usgs
from modules.nwm import download_nwm
from modules.pairs import pair_nwm_usgs

def main() -> None:
    """Main."""
    root = Path("/ised/nwm_explorer_data")

    # download_usgs(
    #     start=pd.Timestamp("2023-09-28"),
    #     end=pd.Timestamp("2025-10-11"),
    #     root=root
    # )

    # download_nwm(
    #     start=pd.Timestamp("2023-10-01"),
    #     end=pd.Timestamp("2025-09-30"),
    #     root=root,
    #     jobs=18,
    #     retries=1
    # )

    # pair_nwm_usgs(
    #     root=root,
    #     start_date=pd.Timestamp("2023-10-01"),
    #     end_date=pd.Timestamp("2025-09-30")
    # )

    evaluations = {
        "FY2024Q1": (pd.Timestamp("2023-10-01"), pd.Timestamp("2023-12-31T23:59")),
        "FY2024Q2": (pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31T23:59")),
        "FY2024Q3": (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-30T23:59")),
        "FY2024Q4": (pd.Timestamp("2024-07-01"), pd.Timestamp("2024-09-30T23:59")),
        "FY2025Q1": (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-12-31T23:59")),
        "FY2025Q2": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31T23:59")),
        "FY2025Q3": (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30T23:59")),
        "FY2025Q4": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30T23:59")),
        "FY2024": (pd.Timestamp("2023-10-01"), pd.Timestamp("2024-09-30T23:59")),
        "FY2025": (pd.Timestamp("2024-10-01"), pd.Timestamp("2025-09-30T23:59")),
        "FY2024-FY2025": (pd.Timestamp("2023-10-01"), pd.Timestamp("2025-09-30T23:59"))
    }

    for l, (s, e) in evaluations.items():
        evaluate(
            label = l,
            root = root,
            start_time = s,
            end_time = e,
            processes = 18,
            sites_per_chunk = 500
        )
    # evals = load_metrics(
    #     root=root,
    #     label="FY2024Q3",
    #     configuration=ModelConfiguration.MEDIUM_RANGE_MEM_1,
    #     metric=Metric.KLING_GUPTA_EFFICIENCY
    # )
    # print(evals)

if __name__ == "__main__":
    main()
