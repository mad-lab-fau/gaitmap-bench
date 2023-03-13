import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union

import gaitmap_challenges.config as challenge_config

HERE = Path(__file__).parent

DEFAULT_RESULTS_DIR = HERE.parent.parent / Path("results")
MAIN_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_CONFIG_FILE = MAIN_REPO_ROOT / Path(".dev_config.json")


@dataclass(frozen=True)
class BenchLocalConfig(challenge_config.LocalConfig):
    results_dir: Path = DEFAULT_RESULTS_DIR


def set_config(config_obj_or_path: Optional[Union[str, Path, BenchLocalConfig]] = None, debug: bool = True):
    if debug is False:
        # In case we actually run results, it is not allowed to set the config via anything else than a env variable.
        if config_obj_or_path is not None:
            warnings.warn(
                "Config can only be set via environment variable (or by using the default dev-config) in non-debug "
                "mode. "
                "At the moment you are trying to set it via a config object or path in `set_config`. "
                "This will be ignored and we fallback to the environment variable. "
                "Please remove, the manual config setting (`set_config(None)`) before submitting your script to the "
                "github repo."
            )
            config_obj_or_path = None

    config_obj = challenge_config.set_config(
        config_obj_or_path, debug, _config_type=BenchLocalConfig, _default_config_file=DEFAULT_CONFIG_FILE
    )
    if debug is False:
        # In this case we need to make sure that the results dir is set to the default one.
        # We will do that silently to avoid having a warning every time, as users will have a local version in their
        # config for debugging purposes.
        if config_obj.results_dir != DEFAULT_RESULTS_DIR:
            config_obj = replace(config_obj, results_dir=DEFAULT_RESULTS_DIR)
            challenge_config.reset_config()
            config_obj = challenge_config.set_config(config_obj, debug)
    return config_obj


def create_config_template(path: Union[str, Path]):
    challenge_config.create_config_template(path, BenchLocalConfig)


# We just reexport that to change the type hint
def config() -> BenchLocalConfig:
    return challenge_config.config()  # type: ignore


# We reexport some of the functions from gaitmap_challenges.config for convenience
reset_config = challenge_config.reset_config

__all__ = ["set_config", "BenchLocalConfig", "config", "reset_config", "create_config_template", "DEFAULT_CONFIG_FILE"]
