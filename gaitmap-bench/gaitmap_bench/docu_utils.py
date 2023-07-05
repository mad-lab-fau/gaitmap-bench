from typing import Dict


def glue_bokeh_md(fig_name: str):
    """Create markdown for bokeh figure."""
    return f":::{{glue:figure}} {fig_name}\n:::"


def tabs(content: Dict[str, str], class_str: str = "", sync: bool = False):
    start = ":::::{tab-set}"
    if class_str:
        start += f"\n:class: {class_str}\n"
    end = ":::::"

    tab_item_template = "::::{{tab-item}} {title}\n{content}\n::::"

    if sync:

        def content_processor(title: str, content_str: str):
            return f":sync: {title}\n\n{content_str}\n"

    else:

        def content_processor(_: str, content_str: str):
            return content_str

    tab_items = [
        tab_item_template.format(title=title, content=content_processor(title, content))
        for title, content in content.items()
    ]

    return "\n".join([start, *tab_items] + [end])
