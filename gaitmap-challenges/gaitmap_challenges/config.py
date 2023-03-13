import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Union, Optional, Type, TypeVar, Any

from gaitmap_datasets import DatasetsConfig
from gaitmap_datasets import set_config as set_datasets_config
from gaitmap_datasets import reset_config as reset_datasets_config

_CONFIG_ENV_VAR: str = "GAITMAP_CHALLENGES_CONFIG"
_GLOBAL_CONFIG: Optional["LocalConfig"] = None
_DEBUG: bool = True

_ConfigT = TypeVar("_ConfigT", bound="LocalConfig")

# Note: Frozen dataclasses can be "overwritten" by using the `replace` method.
@dataclass(frozen=True)
class LocalConfig:
    tmp_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None
    results_dir: Optional[Path] = None
    n_jobs: int = 1
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

    def __post_init__(self):
        path_fields = ("tmp_dir", "cache_dir", "results_dir")
        for field in path_fields:
            if val := getattr(self, field, None) is not None:
                object.__setattr__(self, field, Path(val))


def set_config(
    config_obj_or_path: Optional[Union[str, Path, _ConfigT]] = None,
    debug: bool = True,
    _config_type: Type[_ConfigT] = LocalConfig,
    _default_config_file: Optional[Union[str, Path]] = None,
) -> _ConfigT:
    """Load the config file."""
    if config_obj_or_path is None:
        # Get the config file from the environment variable
        config_obj_or_path = os.environ.get(_CONFIG_ENV_VAR, None)
        if config_obj_or_path is None:
            # In this case we check the default config file
            if _default_config_file is not None and Path(_default_config_file).exists():
                config_obj_or_path = _default_config_file
            else:
                raise ValueError(
                    "Could not load the config!"
                    f"We tried the following things:\n\n"
                    "Config file path -> not specified\n"
                    f"environment variable ({_CONFIG_ENV_VAR}) -> not set\n"
                    f"default config file ({_default_config_file})-> not specified or does not exist"
                )
    if isinstance(config_obj_or_path, (str, Path)):
        config_obj = _config_type.from_json_file(config_obj_or_path)
    elif isinstance(config_obj_or_path, _config_type):
        config_obj = config_obj_or_path
    else:
        raise ValueError("Invalid config object or path.")
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is not None:
        raise ValueError("Config already set.")
    _GLOBAL_CONFIG = config_obj
    set_datasets_config(config_obj.datasets)
    return config_obj


def create_config_template(path: Union[str, Path], _config_type: Type[_ConfigT] = LocalConfig):
    """Create a template json file that can be used to configure the datasets paths.

    Use that method once to create your local config file.

    Then you can use `set_config(path_to_config)` to set the global config to your local config file.

    Parameters
    ----------
    path : Union[str, Path]
        The path to the file where the config should be created.

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
                k.name: sanitize_path(k.default) for k in fields(_config_type) if k.name != "datasets"
            },
            "datasets": {k.name: sanitize_path(k.default) for k in fields(DatasetsConfig)},
        }
        json.dump(config_dict, f, indent=4, sort_keys=True)

    print(f"Created config template at {path.resolve()}.")


def reset_config():
    """Reset the global config to None.

    Afterwards you can use `set_config` to set a new config (e.g. to change the config file during runtime).
    """
    global _GLOBAL_CONFIG  # pylint: disable=global-statement
    _GLOBAL_CONFIG = None
    reset_datasets_config()


def config() -> LocalConfig:
    """Get the global config object."""
    if _GLOBAL_CONFIG is None:
        raise ValueError("Config not set.")
    return _GLOBAL_CONFIG


__all__ = ["set_config", "reset_config", "config", "create_config_template", "LocalConfig"]
