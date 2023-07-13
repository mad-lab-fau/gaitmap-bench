import json
import os
import warnings
from contextlib import suppress
from dataclasses import asdict, dataclass, fields
from os.path import relpath
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, TypedDict, TypeVar, Union

from gaitmap_datasets import DatasetsConfig
from gaitmap_datasets import reset_config as reset_datasets_config
from gaitmap_datasets import set_config as set_datasets_config
from tpcp.parallel import register_global_parallel_callback

_CONFIG_ENV_VAR: str = "GAITMAP_CHALLENGES_CONFIG"
_DEBUG_ENV_VAR: str = "GAITMAP_CHALLENGES_DEBUG"
_GLOBAL_CONFIG: Optional["LocalConfig"] = None
_DEBUG: Optional[bool] = None

_ConfigT = TypeVar("_ConfigT", bound="LocalConfig")


# Note: Frozen dataclasses can be "overwritten" by using the `replace` method.
@dataclass(frozen=True)
class LocalConfig:
    tmp_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    n_jobs: Union[int, str] = 1
    datasets: Optional[DatasetsConfig] = None

    @classmethod
    def from_json_file(cls, config_file: Union[str, Path]):
        """Get config data from json file."""
        with Path(config_file).open(encoding="utf8") as f:
            json_config = json.load(f)
            config_dict = {
                **json_config["gaitmap_challenges"],
                "datasets": DatasetsConfig.from_json_file(config_file),
            }
            return cls(**config_dict)

    def __post_init__(self):  # noqa: D105
        path_fields = ("tmp_dir", "cache_dir", "results_dir")
        for field in path_fields:
            if (val := getattr(self, field, None)) is not None:
                object.__setattr__(self, field, Path(val))

    def _register_dataset_config(self):
        # Register the dataset config
        if self.datasets:
            set_datasets_config(self.datasets)

    def to_json_dict(self, path_relative_to: Optional[Path] = None) -> Dict[str, Any]:
        """Get the config as json-serializable dict.

        Note: This is not meant for round-trip serialization!
        """
        config_dict = asdict(self)
        for name, value in config_dict.items():
            if isinstance(value, Path):
                if path_relative_to is None:
                    config_dict[name] = str(value)
                else:
                    config_dict[name] = str(relpath(value, path_relative_to))

        if self.datasets:
            dataset_config = asdict(self.datasets)
            for name, value in dataset_config.items():
                if isinstance(value, Path):
                    if path_relative_to is None:
                        dataset_config[name] = str(value)
                    else:
                        dataset_config[name] = str(relpath(value, path_relative_to))

            config_dict["datasets"] = dataset_config

        return config_dict


def set_config(
    config_obj_or_path: Optional[Union[str, Path, _ConfigT]] = None,
    debug: Optional[bool] = None,
    _config_type: Type[_ConfigT] = LocalConfig,
    _default_config_file: Optional[Union[str, Path]] = None,
) -> _ConfigT:
    """Load the config file."""
    if _default_config_file is not None and not Path(_default_config_file).exists():
        _default_config_file = None
    look_up_order = (
        config_obj_or_path,
        os.environ.get(_CONFIG_ENV_VAR, None),
        _default_config_file,
    )
    for config_obj_or_path in look_up_order:
        if config_obj_or_path is not None:
            break
    else:
        raise ValueError(
            "Could not load the config! "
            f"We tried the following things:\n\n"
            "Config file path -> not specified\n"
            f"environment variable ({_CONFIG_ENV_VAR}) -> not set\n"
            f"default config file ({_default_config_file}) -> not specified or does not exist"
        )

    if isinstance(config_obj_or_path, (str, Path)):
        config_obj = _config_type.from_json_file(config_obj_or_path)
    elif isinstance(config_obj_or_path, _config_type):
        config_obj = config_obj_or_path
    else:
        raise ValueError("Invalid config object or path.")  # noqa: TRY004
    global _GLOBAL_CONFIG  # noqa: PLW0603
    if _GLOBAL_CONFIG is not None:
        raise ValueError("Config already set.")
    _GLOBAL_CONFIG = config_obj
    config_obj._register_dataset_config()
    global _DEBUG  # noqa: PLW0603
    if _DEBUG is not None:
        raise ValueError("Debug already set.")
    _DEBUG = _resolve_debug(debug)
    return config_obj


def _resolve_debug(debug: Optional[bool]):
    # Debug from environment variable has precedence over direct setting.
    debug_from_env = None
    if os.environ.get(_DEBUG_ENV_VAR, None) is not None:
        debug_from_env = os.environ.get(_DEBUG_ENV_VAR, None).lower() in ("true", "1")
    if debug_from_env is not None:
        if debug is not None:
            warnings.warn(
                f"You specified `debug={debug}` directly via `set_config`, but the environmental variable "
                f"{_DEBUG_ENV_VAR} is also set. "
                f"The configuration from the environmental variable ({debug_from_env=}) will be used! "
                "We recommend removing `debug` from the `set_config` call when you are using the environmental "
                "variable.",
                stacklevel=2,
            )
        debug = debug_from_env
    if debug is None:
        debug = True
    return debug


def create_config_template(
    path: Union[str, Path], _config_type: Type[_ConfigT] = LocalConfig
):
    """Create a template json file that can be used to configure the datasets paths.

    Use that method once to create your local config file.

    Then you can use `set_config(path_to_config)` to set the global config to your local config file.

    Parameters
    ----------
    path : Union[str, Path]
        The path to the file where the config should be created.
    _config_type : Type[_ConfigT], optional
        In case you have a custom config structure, you can pass it here, by default LocalConfig.
        Note, that we don't support arbitrary config structures.
        So this should be considered an internal parameter with no real use outside gaitmap-challenges/bench.

    """
    path = Path(path)
    if path.exists():
        raise ValueError(f"Config file {path} already exists.")

    def sanitize_path(p: Any) -> str:
        if isinstance(p, Path):
            return str(p.resolve())
        return p

    with path.open("w", encoding="utf8") as f:
        config_dict = {
            "gaitmap_challenges": {
                k.name: sanitize_path(k.default)
                for k in fields(_config_type)
                if k.name != "datasets"
            },
            "datasets": {
                k.name: sanitize_path(k.default) for k in fields(DatasetsConfig)
            },
        }
        json.dump(config_dict, f, indent=4, sort_keys=True)

    print(f"Created config template at {path.resolve()}.")


def reset_config():
    """Reset the global config to None.

    Afterwards you can use `set_config` to set a new config (e.g. to change the config file during runtime).
    """
    global _GLOBAL_CONFIG  # noqa: PLW0603
    _GLOBAL_CONFIG = None
    reset_datasets_config()

    global _DEBUG  # noqa: PLW0603
    _DEBUG = None


def config() -> LocalConfig:
    """Get the global config object."""
    if _GLOBAL_CONFIG is None:
        raise ValueError("Config not set.")
    return _GLOBAL_CONFIG


def is_debug_run() -> Optional[bool]:
    """Check if the current run is a debug run."""
    return _DEBUG


def is_config_set() -> bool:
    """Check if the config is set."""
    return _GLOBAL_CONFIG is not None


# This callback (and the register afterwards), works together with a tpcp parallel "hack" that ensures that the config
# is restored in a worker process spawned by joblib.
# This will only have an effect if the config is set in the main process and the parallel implementation is using the
# modified `delayed` function from tpcp.parallel.
class _RestoreConfig(TypedDict):
    config_obj_or_path: LocalConfig
    debug: Optional[bool]


def _config_restore_callback() -> (
    Tuple[Optional[_RestoreConfig], Callable[[_RestoreConfig], None]]
):
    def setter(config_obj: _RestoreConfig):
        reset_config()
        # We set the config manually here to skip the check in set_config.
        # We don't need this, as this method is only called if the config is set in the main process.
        global _DEBUG  # noqa: PLW0603
        _DEBUG = config_obj["debug"]
        global _GLOBAL_CONFIG  # noqa: PLW0603
        _GLOBAL_CONFIG = config_obj["config_obj_or_path"]
        with suppress(AttributeError):
            set_datasets_config(config_obj["config_obj_or_path"].datasets)

    try:
        returned_config = config()
    except ValueError:
        return None, lambda _: None
    return {"config_obj_or_path": returned_config, "debug": is_debug_run()}, setter


register_global_parallel_callback(_config_restore_callback)


__all__ = [
    "set_config",
    "reset_config",
    "config",
    "create_config_template",
    "LocalConfig",
    "is_debug_run",
    "is_config_set",
]
