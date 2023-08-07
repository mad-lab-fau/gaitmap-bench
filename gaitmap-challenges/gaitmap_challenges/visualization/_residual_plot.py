from dataclasses import dataclass
from itertools import chain
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from matplotlib import pyplot as plt
from matplotlib import transforms
from scipy import stats

from gaitmap_challenges._utils import _ensure_label_tuple


def _handle_single_or_list(
    values: Dict[str, Union[List[float], List[List[float]]]]
) -> Tuple[Dict[str, List[float]], Literal["single", "list"]]:
    """Flatten potential nested list."""
    results = {}
    value_type = "single"
    for k, v in values.items():
        if isinstance(v[0], float):
            results[k] = list(v)
        else:
            results[k] = [item for sublist in v for item in sublist]
            value_type = "list"
    return results, value_type


def _prepare_residual_plot_data(cv_result, *, prediction_col_name, reference_col_name):
    labels = cv_result["test_data_labels"].explode().map(_ensure_label_tuple)
    predictions, predictions_type = _handle_single_or_list(cv_result[f"test_single_{prediction_col_name}"])
    references, references_type = _handle_single_or_list(cv_result[f"test_single_{reference_col_name}"])

    assert predictions_type == references_type, "Predictions and references must be of the same type."

    if predictions_type == "list":
        # TODO: Implement this
        raise AssertionError()
    return pd.DataFrame(
        {
            "reference": list(chain(*references.values())),
            "predictions": list(chain(*predictions.values())),
            "label": labels.to_list(),
            "fold": labels.index.to_list(),
        }
    )


def blandaltman_stats(*, reference, prediction, x_val: Literal["mean", "reference", "prediction"] = "mean"):
    if x_val == "mean":
        x = (reference + prediction) / 2
    elif x_val == "reference":
        x = reference.copy()
    elif x_val == "prediction":
        x = prediction.copy()
    else:
        raise ValueError("x_val must be one of `mean`, `reference`, `prediction`.")
    return x, prediction - reference


def plot_blandaltman_annotations(
    error,
    ax=None,
    agreement=1.96,
    confidence=0.95,
):
    """Add annotations (mean and confidence interval) to a blandaltman style plot.

    Code modified based on penguin

    Parameters
    ----------
    error
        values typically plotted on the y-axis of the blandaltma plot
    agreement : float
        Multiple of the standard deviation to draw confidenc interval line
    confidence : float
        CIs for the limits of agreement and the mean
    ax : matplotlib axes
        Axis on which to draw the plot.

    Returns
    -------
    ax : Matplotlib Axes instance
        Returns the Axes object with the plot for further tweaking.

    """
    if ax is None:
        ax = plt.gca()

    # Calculate mean, STD and SEM of x - y
    n = error.size
    dof = n - 1
    mean_diff = np.mean(error)
    std_diff = np.std(error, ddof=1)
    mean_diff_se = np.sqrt(std_diff**2 / n)
    # Limits of agreements
    high = mean_diff + agreement * std_diff
    low = mean_diff - agreement * std_diff
    high_low_se = np.sqrt(3 * std_diff**2 / n)

    # limits of agreement
    ax.axhline(mean_diff, color="k", linestyle="-", lw=2)
    ax.axhline(high, color="k", linestyle=":", lw=1.5)
    ax.axhline(low, color="k", linestyle=":", lw=1.5)

    # Annotate values
    loa_range = high - low
    offset = (loa_range / 100.0) * 1.5
    trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    xloc = 0.98
    ax.text(xloc, mean_diff + offset, "Mean", ha="right", va="bottom", transform=trans)
    ax.text(xloc, mean_diff - offset, "%.2f" % mean_diff, ha="right", va="top", transform=trans)
    ax.text(xloc, high + offset, "+%.2f SD" % agreement, ha="right", va="bottom", transform=trans)
    ax.text(xloc, high - offset, "%.2f" % high, ha="right", va="top", transform=trans)
    ax.text(xloc, low - offset, "-%.2f SD" % agreement, ha="right", va="top", transform=trans)
    ax.text(xloc, low + offset, "%.2f" % low, ha="right", va="bottom", transform=trans)

    # Add 95% confidence intervals for mean bias and limits of agreement
    assert 0 < confidence < 1
    ci = {}
    ci["mean"] = stats.t.interval(confidence, dof, loc=mean_diff, scale=mean_diff_se)
    ci["high"] = stats.t.interval(confidence, dof, loc=high, scale=high_low_se)
    ci["low"] = stats.t.interval(confidence, dof, loc=low, scale=high_low_se)
    ax.axhspan(ci["mean"][0], ci["mean"][1], facecolor="tab:grey", alpha=0.2)
    ax.axhspan(ci["high"][0], ci["high"][1], facecolor="tab:blue", alpha=0.2)
    ax.axhspan(ci["low"][0], ci["low"][1], facecolor="tab:blue", alpha=0.2)

    return ax


def residual_plot_matplotlib(
    cv_results,
    *,
    prediction_col_name,
    reference_col_name,
    metric_name: Optional[str] = None,
    ax=None,
):
    """Create a residual plot using matplotlib.

    For more details on the parameters, see :class:`~ResidualPlot`.
    """
    if ax is None:
        ax = plt.gca()

    df = _prepare_residual_plot_data(
        cv_results,
        prediction_col_name=prediction_col_name,
        reference_col_name=reference_col_name,
    )

    x, y = blandaltman_stats(prediction=df["predictions"], reference=df["reference"], x_val="reference")
    ax.scatter(x, y)
    plot_blandaltman_annotations(y, ax=ax)

    x_label = "Reference"
    if metric_name is not None:
        x_label += f" {metric_name}"
    ax.set_xlabel(x_label)

    y_label = "Prediction - Reference"
    if metric_name is not None:
        y_label += f" ({metric_name})"
    ax.set_ylabel(y_label)

    return ax


def residual_plot_bokeh(
    cv_result,
    *,
    prediction_col_name,
    reference_col_name,
    metric_name: Optional[str] = None,
):
    """Create a residual plot using bokeh.

    For more details on the parameters, see :class:`~ResidualPlot`.
    """
    from bokeh.models import Label, Span
    from bokeh.plotting import figure

    df = _prepare_residual_plot_data(
        cv_result,
        prediction_col_name=prediction_col_name,
        reference_col_name=reference_col_name,
    )

    df["x"], df["y"] = blandaltman_stats(prediction=df["predictions"], reference=df["reference"], x_val="reference")
    p = figure(sizing_mode="stretch_width")
    points = p.scatter("x", "y", source=df, size=10, alpha=0.7)

    label_tooltip = [("dp", "@label"), ("fold", "@fold")]
    my_hover = HoverTool(
        renderers=[points],
        tooltips=[
            *label_tooltip,
        ],
        point_policy="follow_mouse",
    )
    p.add_tools(my_hover)

    x, y = df["x"], df["y"]
    p.add_layout(Span(location=np.mean(y), dimension="width", line_color="black", line_width=2))
    p.add_layout(
        Span(
            location=np.mean(y) + 1.96 * np.std(y),
            dimension="width",
            line_color="black",
            line_dash="dashed",
            line_width=2,
        )
    )
    p.add_layout(
        Span(
            location=np.mean(y) - 1.96 * np.std(y),
            dimension="width",
            line_color="black",
            line_dash="dashed",
            line_width=2,
        )
    )
    label_pos = np.max(x)
    offset = 10
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y),
            y_offset=offset,
            text="Mean",
            text_baseline="bottom",
            text_align="left",
            text_font_size="12pt",
        )
    )
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y),
            y_offset=-offset,
            text=f"{np.mean(y):.2}",
            text_baseline="top",
            text_align="left",
            text_font_size="12pt",
        )
    )
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y) + 1.96 * np.std(y),
            text="+1.96 SD",
            y_offset=offset,
            text_baseline="bottom",
            text_align="left",
            text_font_size="12pt",
        )
    )
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y) + 1.96 * np.std(y),
            text=f"{np.mean(y) + 1.96 * np.std(y):.2}",
            y_offset=-offset,
            text_baseline="top",
            text_align="left",
            text_font_size="12pt",
        )
    )
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y) - 1.96 * np.std(y),
            text="-1.96 SD",
            y_offset=-offset,
            text_baseline="top",
            text_align="left",
            text_font_size="12pt",
        )
    )
    p.add_layout(
        Label(
            x=label_pos,
            y=np.mean(y) - 1.96 * np.std(y),
            text=f"{np.mean(y) - 1.96 * np.std(y):.2}",
            y_offset=offset,
            text_baseline="bottom",
            text_align="left",
            text_font_size="12pt",
        )
    )

    x_label = "Reference"
    if metric_name is not None:
        x_label += f" {metric_name}"
    p.xaxis.axis_label = x_label

    y_label = "Prediction - Reference"
    if metric_name is not None:
        y_label += f" ({metric_name})"
    p.yaxis.axis_label = y_label

    return p


@dataclass
class ResidualPlot:
    """Plot a residual (similar to Blant-Altman) plot to visualize the error dependency.

    Most parameters are shared between both plotting backends.
    Specific parameters are prefixed with `bokeh_` or `matplotlib_` or are expected to be passed to the method directly.

    To use this visualization, you need to make the "raw" predictions available in the cv_result.
    This is usually done by including the with the :class:`~tpcp.validate.NoAgg` wrapper in the scorer output.

    Parameters
    ----------
    cv_result
        The CV results of a single algorithm as loaded by `load_run`.
    prediction_col_name : str
        The name of the column in the cv_results that contains the predictions.
    reference_col_name : str
        The name of the column in the cv_results that contains the reference values.
    metric_name : str, optional
        An optional display name for the used metric. If not provided, the column name is used.

    """

    cv_result: pd.DataFrame
    prediction_col_name: str
    reference_col_name: str
    metric_name: Optional[str] = None

    def bokeh(self):
        """Create the plot using bokeh.

        This creates a plot object that can be displayed using `bokeh.plotting.show`.
        """
        return residual_plot_bokeh(
            self.cv_result,
            prediction_col_name=self.prediction_col_name,
            reference_col_name=self.reference_col_name,
            metric_name=self.metric_name,
        )

    def matplotlib(self, ax=None):
        """Create the plot using matplotlib.

        You can optionally pass an existing matplotlib axes object to plot into.
        """
        return residual_plot_matplotlib(
            self.cv_result,
            prediction_col_name=self.prediction_col_name,
            reference_col_name=self.reference_col_name,
            metric_name=self.metric_name,
            ax=ax,
        )
