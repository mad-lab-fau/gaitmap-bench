from itertools import chain
from typing import Callable, Iterable, List, TypeVar

shared_metadata = {
    "references": ["https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-021-00883-7"],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["Nils Roth et al."],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap/tree/master/gaitmap_mad/gaitmap_mad/"
    "stride_segmentation/hmm",
}

default_metadata = {
    "short_description": "Hierarchical Hidden Markov Model for gait segmentation.",
    "long_description": "This method uses an HMM to segment gait cycles. "
    "The same model as in the original publication is used and all default hyper-parameters are used. ",
    **shared_metadata,
}

retrained_metadata = {
    "short_description": "Hierarchical Hidden Markov Model for gait segmentation with retrained model.",
    "long_description": "This method uses an HMM to segment gait cycles. "
    "The model is specifically retrained for the dataset, but all hyper-parameters remained at there default values. ",
    **shared_metadata,
}


T = TypeVar("T")
K = TypeVar("K")


def apply_and_flatten(values: Iterable[T], apply: Callable[[T], K]) -> List[K]:
    return list(chain(*map(apply, values)))
