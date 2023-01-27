from itertools import chain
from typing import Dict, Literal

import pandas as pd
import holoviews as hv
from holoviews import opts
from bokeh.models import HoverTool


def box_plot(cv_results: Dict[str, pd.DataFrame], metric: str, group_by: Literal["fold", "single"]):

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
            labels = v.index
        elif group_by == "single":
            data = list(chain(*v[metric_name]))
            labels = list(chain(*v["test_data_labels"]))
        else:
            # We should never get here
            raise ValueError()

        if isinstance(k, tuple):
            k = "/".join(k)
        all_results[k] = pd.DataFrame({metric: data, "label": labels})

    all_results = (
        pd.concat(all_results, axis=0, names=("name", "old_idx")).reset_index("old_idx", drop=True).reset_index()
    )

    MyHover = HoverTool(tooltips=[("dp", "@label"), (metric, f"@{metric}"),], point_policy="follow_mouse")

    # We create a boxplot with scatter overlay
    boxplot = hv.BoxWhisker(all_results, "name", metric).opts(opts.BoxWhisker(outlier_alpha=0))
    scatter = hv.Scatter(all_results, "name", [metric, "name", "label"]).opts(
        opts.Scatter(jitter=0.2, alpha=0.5, size=6, color="blue", tools=[MyHover], hover_color="red")
    )
    plot = boxplot * scatter

    return plot
