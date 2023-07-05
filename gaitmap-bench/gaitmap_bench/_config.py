import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Union, cast

import gaitmap_challenges.config as challenge_config

MAIN_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_RESULTS_DIR = MAIN_REPO_ROOT / Path("results")
DEFAULT_CONFIG_FILE = MAIN_REPO_ROOT / Path(".dev_config.json")
DEFAULT_ENTRIES_DIR = MAIN_REPO_ROOT / Path("entries")


@dataclass(frozen=True)
class BenchLocalConfig(challenge_config.LocalConfig):
    results_dir: Path = DEFAULT_RESULTS_DIR


def set_config(
    config_obj_or_path: Optional[Union[str, Path, BenchLocalConfig]] = None,
    debug: Optional[bool] = None,
):
    # We manually resolve, how the internal `set_config` will handle the debug flag.
    # We do this, as this method needs to run additional checks depending on if the debug flag is set or not.
    real_debug = challenge_config._resolve_debug(debug)
    if real_debug is False and config_obj_or_path is not None:
        # In case we actually run results, it is not allowed to set the config via anything else than a env variable.
        warnings.warn(
            "Config can only be set via environment variable (or by using the default dev-config) in non-debug "
            "mode. "
            "At the moment you are trying to set it via a config object or path in `set_config`. "
            "This will be ignored and we fallback to the environment variable. "
            "Please remove, the manual config setting (`set_config(None)`) before submitting your script to the "
            "github repo.",
            stacklevel=2,
        )
        config_obj_or_path = None

    config_obj = challenge_config.set_config(
        config_obj_or_path, debug, _config_type=BenchLocalConfig, _default_config_file=DEFAULT_CONFIG_FILE
    )
    if real_debug is False and config_obj.results_dir != DEFAULT_RESULTS_DIR:
        # In this case we need to make sure that the results dir is set to the default one.
        # We will do that silently to avoid having a warning every time, as users will have a local version in their
        # config for debugging purposes.
        warnings.warn(
            "Custom result dir specified. "
            "This is not allowed for non-debug runs. "
            f"Overwriting with default result dir ({DEFAULT_RESULTS_DIR}).",
            stacklevel=2,
        )
        config_obj = replace(config_obj, results_dir=DEFAULT_RESULTS_DIR)
        challenge_config.reset_config()
        config_obj = challenge_config.set_config(config_obj, debug)
    return config_obj


def create_config_template(path: Union[str, Path]):
    challenge_config.create_config_template(path, BenchLocalConfig)


# We just reexport that to change the type hint
def config() -> BenchLocalConfig:
    return cast(BenchLocalConfig, challenge_config.config())


# We reexport some of the functions from gaitmap_challenges.config for convenience
reset_config = challenge_config.reset_config

is_config_set = challenge_config.is_config_set

__all__ = [
    "set_config",
    "BenchLocalConfig",
    "config",
    "reset_config",
    "is_config_set",
    "create_config_template",
    "DEFAULT_CONFIG_FILE",
    "DEFAULT_RESULTS_DIR",
    "DEFAULT_ENTRIES_DIR",
    "MAIN_REPO_ROOT",
]
