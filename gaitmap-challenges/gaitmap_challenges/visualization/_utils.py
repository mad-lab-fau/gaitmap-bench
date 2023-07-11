from typing import List, Tuple, TypeVar, Union, Dict
import seaborn as sns

int_or_str = TypeVar("int_or_str", int, str, float)


def _ensure_label_tuple(label: Union[int_or_str, List[int_or_str], Tuple[int_or_str, ...]]) -> Tuple[int_or_str, ...]:
    if isinstance(label, (int, str, float)):
        return (label,)
    if isinstance(label, list):
        return tuple(label)
    if isinstance(label, tuple):
        return label
    raise TypeError(f"Invalid type for `{label=}`.")


def replace_legend_labels(ax, rename_dict: Dict[str, str]):
    """Replace the legend labels in a matplotlib plot."""

    legend_texts = ax.get_legend().get_texts()
    for text in legend_texts:
        text.set_text(rename_dict.get(text.get_text(), text.get_text()))
