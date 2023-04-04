from itertools import chain
from typing import Dict, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.plotting import figure
from bokeh.transform import jitter


def _prepare_boxplot_data(
    cv_results: Dict[str, pd.DataFrame], metric: str, group_data_by: Literal["fold", "single"]
) -> pd.DataFrame:
    if group_data_by == "fold":
        metric_name = "test_" + metric
    elif group_data_by == "single":
        metric_name = "test_single_" + metric
    else:
        raise ValueError(f"Invalid value for `{group_data_by=}`.")

    all_results = {}
    for k, v in cv_results.items():
        if isinstance(k, tuple):
            k = "/\n".join(k)
        if group_data_by == "fold":
            data = v[metric_name]
            labels = v.index.astype("str")
            all_results[k] = pd.DataFrame({metric: data, "label": labels})
        elif group_data_by == "single":
            data = list(chain(*v[metric_name]))
            labels = v["test_data_labels"].explode().astype(str)
            all_results[k] = pd.DataFrame({metric: data, "label": labels.to_list(), "fold": labels.index.to_list()})
        else:
            # We should never get here
            raise ValueError()

    return pd.concat(all_results, axis=0, names=("name", "old_idx")).reset_index("old_idx", drop=True).reset_index()


def box_plot_matplotlib(
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    group_data_by: Literal["fold", "single"],
    overlay_scatter: bool = True,
    *,
    ax=None,
):
    all_results = _prepare_boxplot_data(cv_results=cv_results, metric=metric, group_data_by=group_data_by)

    if ax is None:
        _, ax = plt.subplots()

    sns.boxplot(
        data=all_results,
        x="name",
        y=metric,
        showfliers=False,
        ax=ax,
    )
    if overlay_scatter:
        sns.swarmplot(
            data=all_results,
            x="name",
            y=metric,
            color="black",
            ax=ax,
        )
    return ax


def box_plot_bokeh(
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    group_data_by: Literal["fold", "single"],
    overlay_scatter: bool = True,
):

    all_results = _prepare_boxplot_data(cv_results=cv_results, metric=metric, group_data_by=group_data_by)

    value_table = all_results.pivot(index="label", columns="name", values=metric)
    box_plot_stats = pd.DataFrame(mpl.cbook.boxplot_stats(value_table, labels=value_table.columns))
    # Prefix the column names with __ to avoid name clashes with metric names
    box_plot_stats = box_plot_stats.add_prefix("__").rename(columns={"__label": "name"})
    # Merge everything back together
    all_results = all_results.merge(box_plot_stats, on="name")

    data = ColumnDataSource(all_results)

    p = figure(
        x_range=all_results["name"].unique(),
        sizing_mode="stretch_width",
    )

    # outlier range
    whisker = Whisker(base="name", upper="__whishi", lower="__whislo", source=data)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    p.vbar("name", 0.7, "__med", "__q3", source=data, line_color="black")
    p.vbar("name", 0.7, "__q1", "__med", source=data, line_color="black")

    lowest_element = all_results["__whislo"].min()
    highest_element = all_results["__whishi"].max()

    # Overlay scatter
    if overlay_scatter:
        points = p.scatter(
            jitter("name", width=0.4, range=p.x_range), metric, source=data, size=5, color="blue", alpha=0.3
        )
        label_tooltip = [("dp", "@label")]
        if group_data_by == "single":
            label_tooltip.append(("fold", "@fold"))

        MyHover = HoverTool(
            renderers=[points],
            tooltips=[
                *label_tooltip,
                (metric, f"@{metric}"),
            ],
            point_policy="follow_mouse",
        )

        lowest_element = min((all_results[metric].min(), lowest_element))
        highest_element = max((all_results[metric].max(), highest_element))
        p.add_tools(MyHover)

    plot_range = highest_element - lowest_element

    p.y_range.start = lowest_element - plot_range * 0.02
    p.y_range.end = highest_element + plot_range * 0.02

    return p
