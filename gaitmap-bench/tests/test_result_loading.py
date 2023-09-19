from gaitmap_bench import is_config_set, config
from gaitmap_bench.docu_utils import set_docs_config
from gaitmap_challenges.results import get_all_result_paths
from gaitmap_challenges.stride_segmentation.egait_segmentation_validation_2014_original_label import Challenge
import pytest


@pytest.fixture(scope="module", autouse=True)
def _config():
    is_config_set() or set_docs_config()


def test_multi_version_results(recwarn):
    assert len(recwarn) == 0
    get_all_result_paths(Challenge, config().results_dir)
    assert len(recwarn) == 0
