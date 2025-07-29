"""Filtering widgets."""
import panel as pn
import polars as pl
import pandas as pd
import numpy as np

pn.extension("tabulator")

COLUMNS: dict[str, str] = {
    "usgs_site_code": "USGS site code",
    "site_name": "Site name",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "HUC": "HUC",
    "drainage_area": "Drainage (sq.mi.)",
    "contributing_drainage_area": "Contrib. (sq.mi.)"
}
"Mapping from column names to pretty strings."

class SiteInformationTable:
    def __init__(self, site_information: pl.LazyFrame):
        # Data source
        self.data = site_information

        # Site info table
        self.output = pn.pane.Placeholder()
        self.info: pd.DataFrame | None = None
        self.area: np.float64 = np.nan
    
    def update(self, usgs_site_code: str) -> None:
        self.info = self.data.filter(
            pl.col("usgs_site_code") == usgs_site_code
        ).collect().to_pandas()

        # Set catchment area
        self.area = self.info[["drainage_area", "contributing_drainage_area"]].min().min()

        url = self.info["monitoring_url"].iloc[0]
        link = pn.pane.Markdown(f'<a href="{url}" target="_blank">Monitoring location</a>')
        
        df = self.info[list(COLUMNS.keys())].rename(
            columns=COLUMNS).transpose().reset_index()
        df.columns = ["Metadata", "Value"]
        self.output.object = pn.Column(
            pn.widgets.Tabulator(
                df,
                show_index=False,
                theme='bootstrap5',
                stylesheets=[":host .tabulator {font-size: 12px;}"],
                widths={"Metadata": 110, "Value": 220}
            ),
            link
        )

    def servable(self) -> pn.pane.Placeholder:
        return self.output
