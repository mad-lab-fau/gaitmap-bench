from dataclasses import dataclass
from itertools import chain
from typing import Callable, Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Whisker
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, jitter

int_or_str = TypeVar("int_or_str", int, str, float)


@dataclass
class SingleMetricBoxplot:
    cv_results: Dict[str, pd.DataFrame]
    metric: str
    use_aggregation: Literal["fold", "single"] = "single"
    overlay_scatter: bool = True
    label_grouper: Optional[Callable[[pd.Series], pd.Series]] = None
    invert_grouping: bool = False

    def bokeh(self):
        return box_plot_bokeh(
            cv_results=self.cv_results,
            metric=self.metric,
            use_aggregation=self.use_aggregation,
            overlay_scatter=self.overlay_scatter,
            label_grouper=self.label_grouper,
            invert_grouping=self.invert_grouping,
        )

    def matplotlib(self, ax: Optional[plt.Axes] = None):
        return box_plot_matplotlib(
            cv_results=self.cv_results,
            metric=self.metric,
            use_aggregation=self.use_aggregation,
            overlay_scatter=self.overlay_scatter,
            label_grouper=self.label_grouper,
            invert_grouping=self.invert_grouping,
            ax=ax,
        )


def group_by_data_label(level: int, include_all: bool = True, force_order: Optional[Sequence[str]] = None):
    def grouper(labels: pd.Series) -> pd.Categorical:
        """Group labels by the data label at the given level.

        This returns a pd.Series mapping the original label to its group label.
        If `include_all` is True, all group labels are repeated with the groupname "all".
        """
        group_labels = labels.apply(lambda label: label[level]).astype(str)
        group_labels.index = labels
        ordered_names = group_labels.unique().tolist() if force_order is None else force_order
        if include_all:
            group_labels = pd.concat([group_labels, pd.Series("all", index=labels)])
            ordered_names.append("all")
        ordered_names = [str(name) for name in ordered_names]
        order = pd.CategoricalDtype(categories=ordered_names, ordered=True)
        return group_labels.astype(order)

    return grouper


def _ensure_label_tuple(label: Union[int_or_str, List[int_or_str], Tuple[int_or_str, ...]]) -> Tuple[int_or_str, ...]:
    if isinstance(label, (int, str, float)):
        return (label,)
    if isinstance(label, list):
        return tuple(label)
    if isinstance(label, tuple):
        return label
    raise TypeError(f"Invalid type for `{label=}`.")


def _prepare_boxplot_data(
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    use_aggregation: Literal["fold", "single"],
    label_grouper: Optional[Callable[[str], str]] = None,
    invert_grouping: bool = False,
) -> pd.DataFrame:
    if use_aggregation == "fold":
        metric_name = "test_" + metric
    elif use_aggregation == "single":
        metric_name = "test_single_" + metric
    else:
        raise ValueError(f"Invalid value for `{use_aggregation=}`.")

    if use_aggregation != "single" and label_grouper is not None:
        raise ValueError(
            "Cannot use `label_grouper` with `use_aggregation='fold'`. "
            "It only makes sense to use it with `use_aggregation='single'`."
        )

    if invert_grouping and not label_grouper:
        raise ValueError("Cannot use `invert_grouping` without `label_grouper`.")

    all_results = {}
    for k, v in cv_results.items():
        if isinstance(k, tuple):
            name = "/\n".join(k)
        else:
            name = k
        if use_aggregation == "fold":
            data = v[metric_name]
            labels = v.index.astype("str")
            all_results[name] = pd.DataFrame({metric: data, "label": labels})
        elif use_aggregation == "single":
            data = list(chain(*v[metric_name]))
            labels = v["test_data_labels"].explode().map(_ensure_label_tuple)
            all_results[name] = pd.DataFrame({metric: data, "label": labels.to_list(), "fold": labels.index.to_list()})
        else:
            # We should never get here
            raise ValueError()

    df = pd.concat(all_results, axis=0, names=("name", "old_idx")).reset_index("old_idx", drop=True).reset_index()
    # Convert label to category to preserve order
    df["label"] = df["label"].astype("category")
    if label_grouper is not None:
        groups = label_grouper(df["label"].drop_duplicates())
        groups.name = "__group"
        df = df.merge(groups, left_on="label", right_index=True)
    return df.reset_index(drop=True)


def box_plot_matplotlib(
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    use_aggregation: Literal["fold", "single"] = "single",
    overlay_scatter: bool = True,
    label_grouper: Optional[Callable[[pd.Series], pd.Series]] = None,
    invert_grouping: bool = False,
    *,
    ax=None,
):
    all_results = _prepare_boxplot_data(
        cv_results=cv_results,
        metric=metric,
        use_aggregation=use_aggregation,
        label_grouper=label_grouper,
        invert_grouping=invert_grouping,
    )

    if ax is None:
        _, ax = plt.subplots()

    if "__group" in all_results.columns:
        hue = "__group"
        hue_order = all_results["__group"].dtype.categories
    else:
        hue = None
        hue_order = None

    x = "name"

    if invert_grouping:
        hue_order = all_results[x].unique()
        x, hue = hue, x

    sns.boxplot(
        data=all_results,
        x=x,
        y=metric,
        showfliers=not overlay_scatter,
        hue=hue,
        hue_order=hue_order,
        ax=ax,
    )
    if overlay_scatter:
        sns.swarmplot(
            data=all_results,
            x=x,
            y=metric,
            color="black",
            hue=hue,
            dodge=True,
            ax=ax,
            legend=False,
            hue_order=hue_order,
            size=3,
        )

    if "__group" in all_results.columns:
        if invert_grouping:
            ax.legend(title="algorithm")
        else:
            ax.legend(title="group")

    ax.set_xlabel(None)
    return ax


def box_plot_bokeh(
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    use_aggregation: Literal["fold", "single"] = "single",
    overlay_scatter: bool = True,
    label_grouper: Optional[Callable[[pd.Series], pd.Series]] = None,
    invert_grouping: bool = False,
):

    all_results = _prepare_boxplot_data(
        cv_results=cv_results, metric=metric, use_aggregation=use_aggregation, label_grouper=label_grouper
    )

    if "__group" in all_results.columns:
        results = {}
        for group, group_results in all_results.groupby("__group"):
            value_table = group_results.pivot(index="label", columns="name", values=metric)
            value_table = (
                pd.DataFrame(mpl.cbook.boxplot_stats(value_table, labels=value_table.columns))
                .add_prefix("__")
                .rename(columns={"__label": "name"})
                .set_index("name")
            )
            results[group] = value_table
        box_plot_stats = (
            pd.concat(
                results,
                names=("__group", "name"),
                axis=0,
            )
            .reset_index()
            .astype({"__group": all_results["__group"].dtype, "name": all_results["name"].dtype})
        )
        if invert_grouping:
            box_plot_stats = box_plot_stats.assign(__factors=lambda df_: list(zip(df_["__group"], df_["name"])))
            sort_order = ["__group", "name"]
        else:
            box_plot_stats = box_plot_stats.assign(__factors=lambda df_: list(zip(df_["name"], df_["__group"])))
            sort_order = ["name", "__group"]

        all_results = all_results.merge(box_plot_stats, on=["__group", "name"]).sort_values(sort_order)
    else:
        value_table = all_results.pivot(index="label", columns="name", values=metric)
        box_plot_stats = pd.DataFrame(mpl.cbook.boxplot_stats(value_table, labels=value_table.columns))
        # Prefix the column names with __ to avoid name clashes with metric names
        box_plot_stats = (
            box_plot_stats.add_prefix("__")
            .rename(columns={"__label": "name"})
            .assign(__factors=lambda df_: df_["name"])
        )
        # Merge everything back together
        all_results = all_results.merge(box_plot_stats, on="name")

    data = ColumnDataSource(all_results)

    try:
        sorted_factors = all_results["__factors"].drop_duplicates().sort_values()
    except TypeError:
        sorted_factors = all_results["__factors"].drop_duplicates().astype(str).sort_values()

    p = figure(
        x_range=FactorRange(factors=sorted_factors),
        sizing_mode="stretch_width",
        y_axis_label=metric,
    )

    # outlier range
    whisker = Whisker(base="__factors", upper="__whishi", lower="__whislo", source=data)
    whisker.upper_head.size = whisker.lower_head.size = 20
    p.add_layout(whisker)

    # quantile boxes
    if "__group" in all_results.columns:
        if invert_grouping:
            colors = all_results["name"].unique()
        else:
            colors = all_results["__group"].dtype.categories
        color = factor_cmap(
            "__factors", palette=Spectral6, factors=colors, start=1, end=2
        )
    else:
        color = "blue"
    p.vbar("__factors", 0.7, "__med", "__q3", source=data, line_color="black", fill_color=color)
    p.vbar("__factors", 0.7, "__q1", "__med", source=data, line_color="black", fill_color=color)

    lowest_element = all_results["__whislo"].min()
    highest_element = all_results["__whishi"].max()

    # Overlay scatter
    if overlay_scatter:
        points = p.scatter(
            jitter("__factors", width=0.4, range=p.x_range),
            metric,
            source=data,
            size=5,
            line_color="black",
            fill_color="white",
            alpha=0.5,
        )
        label_tooltip = [("dp", "@label")]
        if use_aggregation == "single":
            label_tooltip.append(("fold", "@fold"))

        my_hover = HoverTool(
            renderers=[points],
            tooltips=[
                *label_tooltip,
                (metric, f"@{metric}"),
            ],
            point_policy="follow_mouse",
        )

        lowest_element = min((all_results[metric].min(), lowest_element))
        highest_element = max((all_results[metric].max(), highest_element))
        p.add_tools(my_hover)

    plot_range = highest_element - lowest_element

    p.y_range.start = lowest_element - plot_range * 0.02
    p.y_range.end = highest_element + plot_range * 0.02

    if "__group" in all_results.columns:
        p.xaxis.major_label_orientation = "vertical"

    return p
