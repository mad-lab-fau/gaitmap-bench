import contextlib
import inspect
import json
import os
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Type, Union, cast, ClassVar

import numpy as np
import pkg_resources
import sys
import time
from cpuinfo import cpuinfo
from tpcp import BaseTpcpObject, Dataset
from tpcp.optimize import BaseOptimize


@dataclass(repr=False)
class BaseChallenge(BaseTpcpObject):
    run_start_datetime_utc_timestamp_: float = field(init=False)
    run_start_datetime_: str = field(init=False)
    end_start_datetime_utc_timestamp_: float = field(init=False)
    end_start_datetime_: str = field(init=False)
    runtime_: float = field(init=False)
    dataset_: Dataset = field(init=False)

    optimizer: BaseOptimize = field(init=False)

    NAME: ClassVar[str]
    VERSION: ClassVar[str]

    @property
    def _measure_time(self) -> Callable[[], Generator[None, None, None]]:
        @contextlib.contextmanager
        def timer() -> Generator[None, None, None]:
            self.run_start_datetime_utc_timestamp_ = datetime.utcnow().timestamp()
            self.run_start_datetime_ = datetime.now().astimezone().isoformat()
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            self.end_start_datetime_utc_timestamp_ = datetime.utcnow().timestamp()
            self.end_start_datetime_ = datetime.now().astimezone().isoformat()
            self.runtime_ = end_time - start_time

        return timer

    def run(self, optimizer: BaseOptimize):
        raise NotImplementedError()

    def get_core_results(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def save_core_results(self, folder_path):
        raise NotImplementedError()

    @classmethod
    def load_core_results(cls, folder_path) -> Dict[str, Any]:
        """Load the core results from a folder that have been stored using `save_core_results`.

        When implementing this method, make sure that it remains compatible with results saved using older versions
        of the challenge_class.
        """
        raise NotImplementedError()

    @property
    def __version__(self):
        raise NotImplementedError()


def save_run(
    challenge: BaseChallenge,
    entry_name: Union[str, Tuple[str, ...]],
    custom_metadata: Dict[str, Any],
    path: Union[str, Path],
    stored_filenames_relative_to: Optional[Union[str, Path]] = None,
    use_git: bool = True,
):
    # We use the import path of the challenge class as the name of the challenge
    challenge_name = challenge.__class__.__module__ + "." + challenge.__class__.__name__
    challenge_version = challenge.VERSION
    # We calculate a set of internal metadata based on things we can automatically determine from the run
    metadata = {
        "entry_name": entry_name,
        "challenge_name": challenge_name,
        "challenge_version": challenge_version,
        # Ideally this contains some info about the run, as all parameters should be part of the repr
        "repr_challenge": repr(challenge),
        "repr_pipeline": repr(challenge.optimizer),
        "repr_dataset": repr(challenge.dataset_),
        "run_start_datetime_utc": challenge.run_start_datetime_utc_timestamp_,
        "run_start_datetime": challenge.run_start_datetime_,
        "end_start_datetime_utc": challenge.end_start_datetime_utc_timestamp_,
        "end_start_datetime": challenge.end_start_datetime_,
        "runtime": challenge.runtime_,
        "system_info": {
            "python_version": sys.version,
            "python_implementation": sys.implementation.name,
            "packages_info": {
                d.project_name: d.version
                for d in sorted(pkg_resources.working_set, key=lambda x: x.project_name.lower())
            },
            "cpu": {
                k: v for k, v in cpuinfo.get_cpu_info().items() if k not in ["flags", "hz_advertised", "hz_actual"]
            },
        },
    }

    # Get caller filename
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    if stored_filenames_relative_to is not None:
        filename = str(Path(filename).relative_to(stored_filenames_relative_to))
    metadata["caller_file_path"] = filename

    # Get current git commit hash and check if current git repo is dirty
    if use_git:
        import git  # pylint: disable=import-outside-toplevel

        repo = git.Repo(search_parent_directories=True)
        repo_base_dir = repo.git.rev_parse("--show-toplevel")
        if stored_filenames_relative_to is not None:
            repo_base_dir = str(Path(repo_base_dir).relative_to(stored_filenames_relative_to))
        metadata["git_commit_hash"] = repo.head.object.hexsha
        metadata["git_dirty"] = repo.is_dirty()
        metadata["git_base_dir"] = repo_base_dir

    # In the target folder we create the folder structure
    # challenge_name/challenge_version/entry_name/(start_datetime _ unique_id)
    start_datetime_formatted = datetime.fromtimestamp(challenge.run_start_datetime_utc_timestamp_).strftime(
        "%Y%m%d_%H%M%S"
    )
    path = Path(path)
    if isinstance(entry_name, tuple):
        # We create nested folders
        entry_name = os.sep.join(entry_name)

    path = path / challenge_name / challenge_version / entry_name / start_datetime_formatted
    # If the file already exists, we add a number to the end
    i = 0
    while True:
        path = path.parent / f"{path.name}_{i}"
        if not path.exists():
            break
        i += 1
    path.mkdir(parents=True, exist_ok=True)

    # Save the metadata
    with open(path / "custom_metadata.json", "w", encoding="utf8") as f:
        json.dump(custom_metadata, f)
    with open(path / "metadata.json", "w", encoding="utf8") as f:
        json.dump(metadata, f)

    # Save the results
    result_path = path / "results"
    result_path.mkdir(parents=True, exist_ok=True)
    challenge.save_core_results(result_path)

    print(f"Saved run to {path.resolve()}")
    return path


ResultReturn = namedtuple("data_return", ["metadata", "results", "custom_metadata"])


def load_run_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)

    with open(path / "metadata.json", "r", encoding="utf8") as f:
        metadata = json.load(f)

    return metadata

def load_run(challenge_class: Union[Type[BaseChallenge], BaseChallenge], path: Union[str, Path]) -> ResultReturn:
    challenge_class: Type[BaseChallenge] = cast(
        Type[BaseChallenge], challenge_class if isinstance(challenge_class, type) else challenge_class.__class__
    )
    path = Path(path)

    metadata = load_run_metadata(path)

    assert metadata["challenge_name"] == challenge_class.__module__ + "." + challenge_class.__name__

    results = challenge_class.load_core_results(path / "results")

    # laad custom metadata
    with open(path / "custom_metadata.json", "r", encoding="utf8") as f:
        custom_metadata = json.load(f)
    return ResultReturn(metadata, results, custom_metadata)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
