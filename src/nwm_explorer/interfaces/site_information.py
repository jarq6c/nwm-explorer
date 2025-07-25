"""Filtering widgets."""
import panel as pn
import polars as pl

pn.extension("tabulator")

COLUMNS: dict[str, str] = {
    "usgs_site_code": "USGS site code",
    "site_name": "Site name",
    "latitude": "Latitude",
    "longitude": "Longitude",
    "HUC": "HUC",
    "drainage_area": "Drainage area (sq.mi.)",
    "contributing_drainage_area": "Contrib. area (sq.mi.)"
}
"Mapping from column names to pretty strings."

class SiteInformationTable:
    def __init__(self, site_information: pl.LazyFrame):
        # Data source
        self.data = site_information

        # Site info table
        self.output = pn.pane.Placeholder()
    
    def update(self, usgs_site_code: str) -> None:
        data = self.data.filter(
            pl.col("usgs_site_code") == usgs_site_code
        ).collect().to_pandas()

        url = data["monitoring_url"].iloc[0]
        link = pn.pane.Markdown(f'<a href="{url}" target="_blank">Monitoring location</a>')
        
        df = data[list(COLUMNS.keys())].rename(
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
