"""Retrieve USGS annual peak flows by US state."""
from pathlib import Path

import us
import requests
import pandas as pd

STATE_LIST: list[us.states.State] = us.states.STATES + [us.states.PR, us.states.DC]
"""List of US states."""

BASE_URL: str = "https://nwis.waterdata.usgs.gov/nwis/peak?agency_cd=USGS&format=rdb&state_cd="
"""Base URL."""

def main():
    """Main."""
    # Process each state
    dataframes = []
    for state in STATE_LIST:
        # State code
        state_code = state.abbr.lower()

        # Output file
        ofile = Path(f"./{state_code}.tsv")

        # Skip downloading existing, read file
        if ofile.exists():
            print(f"Found {ofile}")
            dataframes.append(pd.read_csv(
                ofile, dtype=str, comment="#", sep="\t"
                ).iloc[1:, :])
            continue

        # URL
        url = BASE_URL + state_code
        print(f"Retrieving {url}")

        # Download
        response = requests.get(url, timeout=3600).text

        # Save
        print(f"Saving {ofile}")
        with ofile.open("w", encoding="utf-8") as fo:
            fo.write(response)

        # Read file
        dataframes.append(pd.read_csv(
            ofile, dtype=str, comment="#"
            ).iloc[1:, :])

    # Concat
    data = pd.concat(dataframes, ignore_index=True)
    idx = pd.to_datetime(data["peak_dt"], errors="coerce").idxmin()
    print(data.iloc[idx, :])

if __name__ == "__main__":
    main()
