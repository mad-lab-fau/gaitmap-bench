from itertools import chain
from typing import Iterable, Callable, List, TypeVar

metadata = {
    "description": "Hierarchical Hidden Markov Model for gait segmentation",
    "citations": [],
    "code_authors": [],
    "algorithm_authors": [],
    "implementation_link": "",
}


T = TypeVar("T")
K = TypeVar("K")


def apply_and_flatten(values: Iterable[T], apply: Callable[[T], K]) -> List[K]:
    return list(chain(*map(apply, values)))
