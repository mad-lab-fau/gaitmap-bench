"""Helper functions to visualize the results of the challenges."""

__all__ = [
    "SingleMetricBoxplot",
    "group_by_data_label",
    "box_plot_matplotlib",
    "box_plot_bokeh",
    "ResidualPlot",
    "residual_plot_matplotlib",
    "residual_plot_bokeh",
    "replace_legend_labels",
]

from gaitmap_challenges.visualization._box_plot import (
    SingleMetricBoxplot,
    box_plot_bokeh,
    box_plot_matplotlib,
    group_by_data_label,
)
from gaitmap_challenges.visualization._residual_plot import ResidualPlot, residual_plot_bokeh, residual_plot_matplotlib
from gaitmap_challenges.visualization._utils import replace_legend_labels
