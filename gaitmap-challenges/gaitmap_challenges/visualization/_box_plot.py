from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bokeh.models import ColumnDataSource, FactorRange, HoverTool, Whisker
from bokeh.palettes import Spectral6
from bokeh.plotting import figure
from bokeh.transform import factor_cmap, jitter

from gaitmap_challenges._utils import _ensure_label_tuple

__all__ = [
    "SingleMetricBoxplot",
    "box_plot_matplotlib",
    "box_plot_bokeh",
    "group_by_data_label",
]


@dataclass
class SingleMetricBoxplot:
    """Create a boxplot for a single metric.

    The plot can be either created with matplotlib or bokeh, by using the respective methods.

    Most parameters are shared between both plotting backends.
    Specific parameters are prefixed with `bokeh_` or `matplotlib_` or are expected to be passed to the method directly.

    Parameters
    ----------
    cv_results
        A list of cv results of multiple algorithms as loaded by `load_run`.
    metric
        The metric to plot.
        This should be the error metric without the "test\\_" and/or "single\\_" prefix.
    use_aggregation
        Whether to use the "fold" or "single" aggregation.
        With "fold" each point in the final plot is a single fold.
        With "single" we pool all datapoint values from all test folds, so that each point in the final plot is one
        datapoint (i.e. one participant)
        In case of single, we look for the metric in the "test_single_<metric>" column.
        In case of fold, we look for the metric in the "test_<metric>" column.
        This further changes what other parameters make sense.
    overlay_scatter
        Whether to overlay a scatterplot on top of the boxplot.
    label_grouper
        A function that returns a group label for each data point.
        This can be used to split the data into multiple groups that are plotted independently to effectively create
        grouped boxplots.
        Most likely you want to use :func:`group_by_data_label` to create this function.
        Note, that this setting only makes sense when `use_aggregation="single"`.
    invert_grouping
        Whether to invert the grouping returned by `label_grouper`.
        This will just change the order of the boxplots and which boxplots are grouped together.
        By default, all groups of one algorithm are plotted next to each other.
        With `invert_grouping=True` all boxplots of one group are plotted next to each other.
    force_order
        A list of algorithm names that specifies the order in which the boxplots are plotted.
        If you want to force the order of your groups, fix the order in the `label_grouper` function.
    matplotlib_boxplot_props
        Additional properties to pass to the `sns.boxplot` function in the matplotlib backend.

    """

    cv_results: Dict[str, pd.DataFrame]
    metric: str
    use_aggregation: Literal["fold", "single"] = "single"
    overlay_scatter: bool = True
    force_order: Optional[Sequence[str]] = None
    label_grouper: Optional[Callable[[pd.Series], pd.Categorical]] = None
    invert_grouping: bool = False
    matplotlib_boxplot_props: Optional[Dict[str, Any]] = None

    def bokeh(self):
        """Create the plot using bokeh.

        This creates a plot object that can be displayed using `bokeh.plotting.show`.
        """
        return box_plot_bokeh(
            cv_results=self.cv_results,
            metric=self.metric,
            use_aggregation=self.use_aggregation,
            force_order=self.force_order,
            overlay_scatter=self.overlay_scatter,
            label_grouper=self.label_grouper,
            invert_grouping=self.invert_grouping,
        )

    def matplotlib(self, ax: Optional[plt.Axes] = None):
        """Create the plot using matplotlib.

        You can optionally pass an existing matplotlib axes object to plot into.
        """
        return box_plot_matplotlib(
            cv_results=self.cv_results,
            metric=self.metric,
            use_aggregation=self.use_aggregation,
            overlay_scatter=self.overlay_scatter,
            force_order=self.force_order,
            label_grouper=self.label_grouper,
            invert_grouping=self.invert_grouping,
            boxplot_props=self.matplotlib_boxplot_props,
            ax=ax,
        )


def group_by_data_label(
    level: Union[int, str],
    include_all: Union[bool, str] = True,
    force_order: Optional[Sequence[str]] = None,
):
    """Create a grouper function that groups labels by the data label at the given level.

    This returns a pd.Series mapping the original label to its group label.

    Parameters
    ----------
    level
        The level to groupby. We assume that the original dataset had multiple levels.
        If an integer is passed, the level is assumed to be the index of the label.
        If a string is passed, the level is assumed to be the name of the level.
    include_all
        If True a group "all" is added that contains all labels.
        To customize the name of the group, pass a string.
    force_order
        If given, the order of the groups is forced to be the given order.
        Note, that this sequence must contain all groups (including "all" if `include_all` is True).

    """

    def grouper(labels: pd.Series) -> pd.Categorical:
        """Group labels by the data label at the given level.

        This returns a pd.Series mapping the original label to its group label.
        If `include_all` is True, all group labels are repeated with the groupname "all".
        """
        if isinstance(level, int):
            group_labels = labels.apply(lambda label: label[level]).astype(str)
        elif isinstance(level, str):
            # In this case we assume that the labels are loaded as namedtuples
            group_labels = labels.apply(lambda label: getattr(label, level)).astype(str)
        else:
            raise TypeError(f"level must be either int or str, but was {type(level)}")

        group_labels.index = labels
        ordered_names = group_labels.unique().tolist()
        if include_all is not False:
            include_all_name = "all" if include_all is True else include_all
            group_labels = pd.concat([group_labels, pd.Series(include_all_name, index=labels)])
            ordered_names.append(include_all_name)
        ordered_names = ordered_names if force_order is None else force_order
        ordered_names = [str(name) for name in ordered_names]
        order = pd.CategoricalDtype(categories=ordered_names, ordered=True)
        return group_labels.astype(order)

    return grouper


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
        name = "/\n".join(k) if isinstance(k, tuple) else k
        if use_aggregation == "fold":
            data = v[metric_name]
            labels = v.index.astype("str")
            all_results[name] = pd.DataFrame({metric: data, "label": labels})
        elif use_aggregation == "single":
            data = list(chain(*v[metric_name]))
            labels = v["test_data_labels"].explode().map(_ensure_label_tuple)
            all_results[name] = pd.DataFrame(
                {
                    metric: data,
                    "label": labels.to_list(),
                    "fold": labels.index.to_list(),
                }
            )
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


def box_plot_matplotlib(  # noqa: PLR0913
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    use_aggregation: Literal["fold", "single"] = "single",
    overlay_scatter: bool = True,
    label_grouper: Optional[Callable[[pd.Series], pd.Categorical]] = None,
    invert_grouping: bool = False,
    force_order: Optional[Sequence[str]] = None,
    boxplot_props: Optional[Dict[str, Any]] = None,
    *,
    ax=None,
):
    """Create a boxplot using matplotlib from the CV results.

    See :class:`~SingleMetricBoxplot` for details on the parameters.
    """
    all_results = _prepare_boxplot_data(
        cv_results=cv_results,
        metric=metric,
        use_aggregation=use_aggregation,
        label_grouper=label_grouper,
        invert_grouping=invert_grouping,
    )

    order = all_results["name"].unique() if force_order is None else force_order

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
        order, hue_order = hue_order, order
        x, hue = hue, x

    sns.boxplot(
        data=all_results,
        x=x,
        y=metric,
        showfliers=not overlay_scatter,
        hue=hue,
        hue_order=hue_order,
        order=order,
        ax=ax,
        **(boxplot_props or {}),
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
            order=order,
            size=3,
        )

    if "__group" in all_results.columns:
        if invert_grouping:
            ax.legend(title="algorithm")
        else:
            ax.legend(title="group")

    ax.set_xlabel(None)
    return ax


def box_plot_bokeh(  # noqa: PLR0913, PLR0915
    cv_results: Dict[str, pd.DataFrame],
    metric: str,
    use_aggregation: Literal["fold", "single"] = "single",
    force_order: Optional[Sequence[str]] = None,
    overlay_scatter: bool = True,
    label_grouper: Optional[Callable[[pd.Series], pd.Categorical]] = None,
    invert_grouping: bool = False,
):
    """Create a boxplot using bokeh from the CV results.

    See :class:`~SingleMetricBoxplot` for details on the parameters.
    """
    all_results = _prepare_boxplot_data(
        cv_results=cv_results,
        metric=metric,
        use_aggregation=use_aggregation,
        label_grouper=label_grouper,
    )

    # We use categorical dtypes to make sorting easier
    if force_order is not None:  # noqa: SIM108
        # We use the order category dtype trick to sort the factors
        name_dtype = pd.CategoricalDtype(categories=force_order, ordered=True)
    else:
        name_dtype = str

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
            .astype(
                {
                    "__group": all_results["__group"].dtype,
                    "name": all_results["name"].dtype,
                }
            )
        )
        sort_order = ["__group", "name"] if invert_grouping else ["name", "__group"]
        sorted_factors = (
            box_plot_stats[sort_order].drop_duplicates().astype({"name": name_dtype}).sort_values(by=sort_order)
        )
        sorted_factors = list(sorted_factors.itertuples(index=False))
        box_plot_stats = box_plot_stats.assign(__factors=lambda df_: list(zip(df_[sort_order[0]], df_[sort_order[1]])))

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

        sorted_factors = box_plot_stats["name"].drop_duplicates().astype(str).astype(name_dtype).sort_values()
        sorted_factors = list(sorted_factors)
        # Merge everything back together
        all_results = all_results.merge(box_plot_stats, on="name")

    data = ColumnDataSource(all_results)

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
        colors = all_results["name"].unique() if invert_grouping else all_results["__group"].dtype.categories
        color = factor_cmap("__factors", palette=Spectral6, factors=colors, start=1, end=2)
    else:
        color = "blue"
    p.vbar(
        "__factors",
        0.7,
        "__med",
        "__q3",
        source=data,
        line_color="black",
        fill_color=color,
    )
    p.vbar(
        "__factors",
        0.7,
        "__q1",
        "__med",
        source=data,
        line_color="black",
        fill_color=color,
    )

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
