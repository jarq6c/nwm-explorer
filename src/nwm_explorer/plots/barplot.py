"""Display standardized bar plots."""
import numpy as np
import numpy.typing as npt
import panel as pn
import plotly.graph_objects as go

class BarPlot:
    def __init__(
            self
            ) -> None:
        self.data = [go.Bar(
            name=""
        )]
        self.layout = go.Layout(
            height=250,
            width=440,
            margin=dict(l=0, r=0, t=0, b=0),
            modebar=dict(
                remove=["lasso", "select", "pan", "autoscale", "zoomin", "zoomout"],
                orientation="v"
            )
        )
        self.figure = dict(data=self.data, layout=self.layout)
        self.pane = pn.pane.Plotly(self.figure)
    
    def update(
            self, 
            xdata: npt.ArrayLike,
            ydata: npt.ArrayLike,
            ydata_lower: npt.ArrayLike,
            ydata_upper: npt.ArrayLike,
            xlabel: str,
            ylabel: str
        ) -> None:
        # Construct custom data
        custom_data = np.hstack((ydata_lower[:, np.newaxis], ydata_upper[:, np.newaxis]))

        # Update trace
        self.data[0].update(dict(
            x=xdata,
            y=ydata,
            customdata=custom_data,
            hovertemplate=(
                f"{xlabel}: " + "%{x}<br>" + 
                f"{ylabel}: " + "%{customdata[0]:.2f} -- %{customdata[1]:.2f} (%{y:.2f})"
            ),
            error_y=dict(
                type="data",
                array=ydata_upper - ydata,
                arrayminus=ydata - ydata_lower
            )
        ))

        # Update axes
        self.layout.update(dict(
            xaxis=dict(title=dict(text=xlabel)),
            yaxis=dict(title=dict(text=ylabel))
        ))
    
    def refresh(self) -> None:
        self.figure.update(dict(data=self.data, layout=self.layout))
        self.pane.object = self.figure
    
    def servable(self) -> pn.Card:
        return pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
