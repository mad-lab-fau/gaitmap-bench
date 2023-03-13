from pathlib import Path
from typing import NamedTuple, Dict, Tuple
import toml


class Entry(NamedTuple):
    name: str
    group_name: str
    base_folder: Path
    setup: str
    command_template: str
    command: str


def find_all_entries(base_folder: Path) -> Dict[str, Dict[str, Entry]]:
    """Find all groups of entries in the given base folder.

    This looks through the folders and searches the `gaitmap_bench.toml` file.
    This marks the base directory of a group of entries.
    It then parses the config file and returns the information.

    Parameters
    ----------
    base_folder: The folder to search for entries in.
    """
    entries = {}
    for entry_file in base_folder.rglob("gaitmap_bench.toml"):
        with entry_file.open("r") as f:
            settings = toml.load(f)

        challenges = settings["challenge"]
        for challenge_type_name, challenge_type in challenges.items():
            for challenge_name, challenge_entries in challenge_type.items():
                for entry_name, entry_settings in challenge_entries.items():
                    entry_group = Entry(
                        name=entry_name,
                        group_name=settings["name"],
                        base_folder=entry_file.parent.relative_to(base_folder),
                        setup=settings["setup"],
                        command_template=settings["command_template"],
                        command=entry_settings,
                    )
                    entries.setdefault(challenge_type_name, {}).setdefault(challenge_name, []).append(entry_group)
    return entries
