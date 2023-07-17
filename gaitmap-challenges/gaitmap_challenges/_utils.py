from typing import List, Tuple, TypeVar, Union

int_or_str = TypeVar("int_or_str", int, str, float)


def _ensure_label_tuple(label: Union[int_or_str, List[int_or_str], Tuple[int_or_str, ...]]) -> Tuple[int_or_str, ...]:
    if isinstance(label, (int, str, float)):
        return (label,)
    if isinstance(label, list):
        return tuple(label)
    if isinstance(label, tuple):
        return label
    raise TypeError(f"Invalid type for `{label=}`.")
