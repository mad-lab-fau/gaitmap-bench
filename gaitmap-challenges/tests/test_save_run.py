import shutil
from pathlib import Path

import pandas as pd
from tpcp import Dataset, Pipeline
from tpcp.optimize import BaseOptimize, DummyOptimize

from gaitmap_challenges import load_run, save_run
from gaitmap_challenges.challenge_base import BaseChallenge


class DummyDataset(Dataset):
    def create_index(self) -> pd.DataFrame:
        return pd.DataFrame({"id": [1, 2, 3]})


class DummyChallenge(BaseChallenge):

    VERSION = "1.0.0"

    def run(self, optimizer: BaseOptimize):
        self.optimizer = optimizer
        self.dataset_ = DummyDataset()
        with self._measure_time():
            pipe = optimizer.optimize(self.dataset_).optimized_pipeline_
            pipe.run(self.dataset_)
            self.dummy_results_ = "dummy_results"
        return self

    @classmethod
    def load_core_results(cls, folder_path, **kwargs):
        folder_path = Path(folder_path)
        with open(folder_path / "dummy_results.txt") as f:
            return {"dummy_results": f.read()}

    def save_core_results(self, folder_path, **kwargs):
        folder_path = Path(folder_path)
        with open(folder_path / "dummy_results.txt", "w") as f:
            f.write(self.dummy_results_)


class DummyPipeline(Pipeline):
    def run(self, datapoint):
        return self


def test_save_load_run(tmp_path):
    challenge = DummyChallenge()
    optimizer = DummyOptimize(
        pipeline=DummyPipeline(),
    )
    challenge.run(optimizer)

    actual_path = save_run(challenge, "test", custom_metadata={"c_meta": 1}, path=tmp_path)

    # Check that path is relative to tmp_path
    assert tmp_path in actual_path.parents
    # Check that all files exists
    assert actual_path.exists()
    assert (actual_path / "metadata.json").exists()
    assert (actual_path / "custom_metadata.json").exists()

    loaded_results = load_run(challenge.__class__, actual_path)
    assert loaded_results.custom_metadata == {"c_meta": 1}
    assert loaded_results.results == {"dummy_results": "dummy_results"}
    assert loaded_results.metadata["entry_name"] == "test"
    assert loaded_results.metadata["challenge_name"] == "tests.test_save_run.DummyChallenge"
    assert loaded_results.metadata["challenge_version"] == "1.0.0"
    assert loaded_results.metadata["is_debug_run"] is True
    # As it is a debug run, the folder name should start with an underscore
    assert actual_path.name.startswith("_")


def test_manual_test():

    # This will create the output in the current directory
    # Set a breakpoint at the last line and inspect the output
    challenge = DummyChallenge()
    optimizer = DummyOptimize(
        pipeline=DummyPipeline(),
    )
    challenge.run(optimizer)

    path = Path(__file__).parent / "_results"
    save_run(challenge, "test", custom_metadata={"c_meta": 1}, path=path)

    shutil.rmtree(path)
