from itertools import chain
from typing import Callable, Iterable, List, TypeVar

metadata = {
    "short_description": "Hierarchical Hidden Markov Model for gait segmentation",
    "long_description": "",
    "references": [],
    "code_authors": [],
    "algorithm_authors": [],
    "implementation_link": "",
}


T = TypeVar("T")
K = TypeVar("K")


def apply_and_flatten(values: Iterable[T], apply: Callable[[T], K]) -> List[K]:
    return list(chain(*map(apply, values)))
