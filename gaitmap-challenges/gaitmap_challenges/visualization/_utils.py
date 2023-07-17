from typing import Dict


def replace_legend_labels(ax, rename_dict: Dict[str, str]):
    """Replace the legend labels in a matplotlib plot."""
    legend_texts = ax.get_legend().get_texts()
    for text in legend_texts:
        text.set_text(rename_dict.get(text.get_text(), text.get_text()))
