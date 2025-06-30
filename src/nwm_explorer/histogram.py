"""Generate standardized histogram plots."""
from dataclasses import dataclass
from typing import Any
import numpy as np
import numpy.typing as npt
import panel as pn
import plotly.graph_objects as go
from plotly.basedatatypes import BaseTraceType

def generate_histogram(
        x: npt.ArrayLike,
        xmin: float,
        xmax: float,
        bin_width: float,
        density: bool = False
    ) -> tuple[npt.NDArray[np.float64], list[str]]:
    nbins = int((xmax - xmin) / bin_width)
    bin_centers = np.linspace(xmin + bin_width / 2, xmax - bin_width / 2, nbins)
    counts = []
    for bc in bin_centers:
        left = bc - bin_width / 2
        right = bc + bin_width / 2
        counts.append(x[(left <= x) & (x <= right)].size)

    below_minimum = x[x < xmin].size
    counts = np.insert(counts, 0, below_minimum)
    bin_centers = np.insert(bin_centers, 0, xmin - bin_width)

    above_maximum = x[x > xmax].size
    counts = np.append(counts, above_maximum)
    bin_centers = np.append(bin_centers, xmax + bin_width)

    if density:
        counts = counts / np.sum(counts)
    return bin_centers, counts

@dataclass
class PlotlyCard:
    data: list[BaseTraceType]
    layout: go.Layout

    def __post_init__(self) -> None:
        # Build pane
        self.pane = pn.pane.Plotly({
            "data": self.data,
            "layout": self.layout
        }, config={'displaylogo': False})

        # Build card
        self.card = pn.Card(
            self.pane,
            collapsible=False,
            hide_header=True
        )
    
    def refresh(self) -> None:
        # Update pane
        self.pane.object = {
            "data": self.data,
            "layout": self.layout
        }
    
    def update_layout(self, parameters: dict[str, Any]) -> None:
        # Update layout
        self.layout.update(parameters)
    
    def update_data(self, parameters: dict[str, Any], index: int = 0) -> None:
        # Update trace
        self.data[index].update(parameters)
    
    def update_config(self, parameters: dict[str, Any]) -> None:
        # Assign config is non-existent
        if self.pane.config is None:
            self.pane.config = parameters
            return
        
        # Update existing config
        self.pane.config.update(parameters)
    
    def update_xaxis_title_text(self, title: str) -> None:
        # Update layout
        self.layout["xaxis"]["title"].update({"text": title})
    
    def servable(self) -> pn.Card:
        return self.card

class HistogramCard:
    def __init__(
            self,
            data: npt.ArrayLike,
            lower: npt.ArrayLike,
            upper: npt.ArrayLike,
            label: str,
            minimum: float = -1.0,
            maximum: float = 1.0,
            bin_width: float = 0.2,
            ytitle: str | None = None,
            density: bool = True
        ):
        # Save parameters
        self.label = label
        self.minimum = minimum
        self.maximum = maximum
        self.bin_width = bin_width
        self.density = density

        # Generate histogram data
        bin_centers, frequencies = generate_histogram(
            data, minimum, maximum, bin_width, density
        )
        _, lo_freq = generate_histogram(
            lower, minimum, maximum, bin_width, density
        )
        _, hi_freq = generate_histogram(
            upper, minimum, maximum, bin_width, density
        )

        # Determine histogram error bars
        estimates = np.vstack((frequencies, lo_freq, hi_freq))
        e_lo = frequencies - np.min(estimates, axis=0)
        e_hi = np.max(estimates, axis=0) - frequencies

        # Number of samples
        samples = len(data)

        # Ticks
        tickvals = [minimum, (minimum+maximum)/2, maximum]
        ticktext = [f"{t:.1f}" for t in tickvals]

        # y-title
        if ytitle is None:
            if self.density:
                ytitle = "Frequency (%)"
            else:
                ytitle = "Count"

        # Convert frequencies
        if self.density:
            frequencies = frequencies * 100
            e_lo = e_lo * 100
            e_hi = e_hi * 100

        # Build plot data with annotations
        plot_data: list[BaseTraceType] = [
            go.Bar(
                x=bin_centers,
                y=frequencies,
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=e_hi,
                    arrayminus=e_lo
                    )
                ),
            go.Scatter(mode="text", textposition="top center"),
            go.Scatter(mode="text", textposition="top center")
            ]
        if frequencies[0] != 0:
            plot_data[1].update(dict(
                x=[bin_centers[0]],
                y=[(frequencies[0]+e_hi[0])*1.05],
                text=[f"<{minimum:.1f}"]
                ))
        if frequencies[-1] != 0:
            plot_data[2].update(dict(
                x=[bin_centers[-1]],
                y=[(frequencies[-1]+e_hi[-1])*1.05],
                text=[f">{maximum:.1f}"]
                ))

        # Build card
        self.card = PlotlyCard(
            plot_data,
            go.Layout(
                dragmode=False,
                hovermode=False,
                showlegend=False,
                height=250,
                width=300,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    range=[minimum-bin_width*2, maximum+bin_width*2],
                    title=dict(text=label + f" ({samples})"),
                    ticks="outside",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickcolor="black",
                    ticklen=6,
                    minor=dict(
                        tickmode="array",
                        tickvals=np.arange(minimum, maximum, bin_width).tolist(),
                        tickcolor="black",
                        ticklen=3
                        )
                    ),
                yaxis=dict(title=dict(text=ytitle))
            )
        )

        # Disable mode bar
        self.card.update_config({"displayModeBar": False})

    def update_data(
            self,
            data: npt.ArrayLike,
            lower: npt.ArrayLike,
            upper: npt.ArrayLike
        ) -> None:
        # Generate histogram data
        bin_centers, frequencies = generate_histogram(
            data, self.minimum, self.maximum, self.bin_width, self.density
        )
        _, lo_freq = generate_histogram(
            lower, self.minimum, self.maximum, self.bin_width, self.density
        )
        _, hi_freq = generate_histogram(
            upper, self.minimum, self.maximum, self.bin_width, self.density
        )

        # Determine histogram error bars
        estimates = np.vstack((frequencies, lo_freq, hi_freq))
        e_lo = frequencies - np.min(estimates, axis=0)
        e_hi = np.max(estimates, axis=0) - frequencies

        # Convert frequencies
        if self.density:
            frequencies = frequencies * 100
            e_lo = e_lo * 100
            e_hi = e_hi * 100

        # Update trace
        self.card.update_data(dict(
            x=bin_centers,
            y=frequencies,
            error_y=dict(
                type="data",
                symmetric=False,
                array=e_hi,
                arrayminus=e_lo
                )
            ))
        if frequencies[0] != 0:
            left_label = f"<{self.minimum:.1f}"
        else:
            left_label = ""
        self.card.update_data(dict(
            x=[bin_centers[0]],
            y=[(frequencies[0]+e_hi[0])*1.05],
            text=[left_label]
            ), 1)
        if frequencies[-1] != 0:
            right_label = f">{self.maximum:.1f}"
        else:
            right_label = ""
        self.card.update_data(dict(
            x=[bin_centers[-1]],
            y=[(frequencies[-1]+e_hi[-1])*1.05],
            text=[right_label]
            ), -1)

        # Update layout
        samples = len(data)
        self.card.update_xaxis_title_text(self.label + f" ({samples})")
    
    def servable(self) -> pn.Card:
        return self.card.servable()
    
    def refresh(self) -> None:
        self.card.refresh()

@dataclass
class HistogramGrid:
    data: list[tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
    specs: list[tuple[float, float, float]]
    labels: list[str]
    columns: int

    def __post_init__(self) -> None:
        self.cards: list[HistogramCard] = []
        for d, l, s in zip(self.data, self.labels, self.specs):
            self.cards.append(HistogramCard(*d, l, *s))

        self.grid = pn.GridBox(*[c.servable() for c in self.cards],
            ncols=self.columns)
    
    def update_data(
            self,
            data: list[tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]]
            ) -> None:
        for d, c in zip(data, self.cards):
            if d is not None:
                c.update_data(*d)
    
    def servable(self) -> pn.GridBox:
        return self.grid
    
    def refresh(self) -> None:
        for c in self.cards:
            c.refresh()
