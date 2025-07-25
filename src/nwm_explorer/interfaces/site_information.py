"""Filtering widgets."""
import panel as pn
import polars as pl

pn.extension("tabulator")

class SiteInformationTable:
    def __init__(self, site_information: pl.LazyFrame):
        # Data source
        self.data = site_information

        # Site info table
        self.output = pn.pane.Markdown("")
    
    def update(self, usgs_site_code: str) -> None:
        data = self.data.filter(
            pl.col("usgs_site_code") == usgs_site_code
        ).collect()
        print(data)
        self.output.object = f"# {usgs_site_code}"

    def servable(self) -> pn.Card:
        return pn.Card(
            self.output,
            title="Site Information",
            collapsible=False,
            width=300,
            height=200
        )
