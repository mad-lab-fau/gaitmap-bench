from itertools import chain
from typing import Dict, Literal

import numpy as np
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, Whisker
from bokeh.plotting import figure
from bokeh.transform import jitter


def box_plot(
    cv_results: Dict[str, pd.DataFrame], metric: str, group_by: Literal["fold", "single"], overlay_scatter: bool = True
):

    if group_by == "fold":
        metric_name = "test_" + metric
    elif group_by == "single":
        metric_name = "test_single_" + metric
    else:
        raise ValueError(f"Invalid value for `group_by`: {group_by}")

    all_results = {}
    for k, v in cv_results.items():
        if group_by == "fold":
            data = v[metric_name]
            labels = v.index.astype("str")
        elif group_by == "single":
            data = list(chain(*v[metric_name]))
            labels = [str(e) for e in chain(*v["test_data_labels"])]
        else:
            # We should never get here
            raise ValueError()

        if isinstance(k, tuple):
            k = "/\n".join(k)
        all_results[k] = pd.DataFrame({metric: data, "label": labels})

    all_results = (
        pd.concat(all_results, axis=0, names=("name", "old_idx")).reset_index("old_idx", drop=True).reset_index()
    )

    # compute quantiles
    qs = all_results.groupby("name")[metric].quantile([0.25, 0.5, 0.75])
    qs = qs.unstack().reset_index()
    qs.columns = ["name", "q1__", "q2__", "q3__"]
    all_results = pd.merge(all_results, qs, on="name", how="left")

    # compute IQR outlier bounds
    iqr = all_results.q3__ - all_results.q1__
    all_results["upper__"] = all_results.q3__ + 1.5 * iqr
    all_results["lower__"] = all_results.q1__ - 1.5 * iqr

    data = ColumnDataSource(all_results)

    p = figure(
        x_range=all_results["name"].unique(),
        sizing_mode="stretch_width",
    )

    # outlier range
    whisker = Whisker(base="name", upper="upper__", lower="lower__", source=data)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    p.vbar("name", 0.7, "q2__", "q3__", source=data, line_color="black")
    p.vbar("name", 0.7, "q1__", "q2__", source=data, line_color="black")

    lowest_element = all_results["lower__"].min()
    highest_element = all_results["upper__"].max()

    # Overlay scatter
    if overlay_scatter:
        points = p.scatter(
            jitter("name", width=0.4, range=p.x_range), metric, source=data, size=5, color="blue", alpha=0.3
        )
        MyHover = HoverTool(
            renderers=[points],
            tooltips=[
                ("dp", "@label"),
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
