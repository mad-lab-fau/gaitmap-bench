import hashlib
from typing import Set, Sequence

import click
from rich.console import Console
from rich.table import Table

from gaitmap_bench import create_config_template
from gaitmap_bench._config import DEFAULT_CONFIG_FILE, DEFAULT_ENTRIES_DIR
from gaitmap_bench._utils import find_all_entries


def _determine_shortest_required_length(hashes: Sequence[str], test_lengths: Sequence[int]) -> int:
    """Determine the shortest length of a hash that is unique for the given set of hashes.

    Parameters
    ----------
    hashes : Sequence[str]
        The set of hashes to test.
    test_lengths : Sequence[int]
        The lengths to test.

    Returns
    -------
    int
        The shortest length that is unique.
        If no length is unique, -1 is returned.
    """
    # Just to be safe we check that there is no collision to begin with
    if len(set(hashes)) != len(hashes):
        raise ValueError("The given hashes are not unique.")
    for length in test_lengths:
        if len(set(h[:length] for h in hashes)) == len(hashes):
            return length

    return len(hashes[0])


@click.group()
def cli():
    """Run and configure the benchmark suite.

    It can be used for the following tasks:

    \b
    - Create a template config for your local configuration.
    - List all benchmarks entries that are registered in the `entries` folder.
    - Run individual benchmarks.
    - (Future) scaffold a new entry to a benchmark.
    """
    pass


@cli.command()
@click.argument("path", type=click.Path(exists=False), default=DEFAULT_CONFIG_FILE)
def create_config(path):
    """Create a template config file for the benchmarking suite."""
    create_config_template(path)


@cli.command()
@click.argument("path", type=click.Path(exists=False), default=DEFAULT_ENTRIES_DIR)
@click.option(
    "--group",
    "-g",
    multiple=True,
    type=str,
    default=None,
    help="Only list entries of the given group. (Can be specified multiple times)",
)
@click.option("--show-command", "-c", is_flag=True, help="Show the command that is used to run the entry.")
def list_entries(path, group, show_command):
    """List all entries that are registered in the `entries` folder."""
    all_entries = find_all_entries(path)
    console = Console()

    if group:
        tmp_entries = {}
        # We sort by length of the name level.
        # This way we make sure that we first include the specific challenges and then the challenge groups
        # To avoid overwriting entries.
        # Not sure if this is really required, but seems to work.
        group_as_parts = sorted([g.split(".") for g in group], key=lambda x: len(x), reverse=True)
        for group in group_as_parts:
            if len(group) == 1:
                if not group[0] in all_entries:
                    raise ValueError(f"Invalid challenge-group name: {group[0]}")
                tmp_entries[group[0]] = all_entries[group[0]]
            elif len(group) == 2:
                challenge_group, challenge = group
                if challenge_group not in all_entries:
                    raise ValueError(f"Invalid challenge-group name: {challenge_group}")
                if challenge not in all_entries[challenge_group]:
                    raise ValueError(f"Invalid challenge name: {challenge}")
                tmp_entries.setdefault(challenge_group, {})[challenge] = all_entries[challenge_group][challenge]
            else:
                raise ValueError(f"The challenge name should only contain one `.` at max. But is is: {group}")

        all_entries = tmp_entries

    # We iterate all group and challenge combinations and create sha1 has them
    challenge_hashes = {}
    challenge_hashes_list = []
    for group_name, group_entries in all_entries.items():
        for challenge_name, entries in group_entries.items():
            name_hash = hashlib.sha1(f"{group_name}.{challenge_name}".encode("utf-8")).hexdigest()
            challenge_hashes.setdefault(group_name, {})[challenge_name] = name_hash
            challenge_hashes_list.append(name_hash)

    # We determine the shortest length of the hash that is unique
    challenge_hash_length = _determine_shortest_required_length(challenge_hashes_list, [3, 6, 9])
    del challenge_hashes_list

    for group_name, group_entries in all_entries.items():
        for challenge_name, entries in group_entries.items():

            # We hash all entry names and determine the shortest length of the hash that is unique
            entry_hashes = {
                (name := f"{entry.group_name}.{entry.name}"): hashlib.sha1(name.encode("utf-8")).hexdigest()
                for entry in entries
            }
            entry_hash_length = _determine_shortest_required_length(list(entry_hashes.values()), [3, 6, 9])

            table = Table(title=f"{group_name}.{challenge_name}")
            table.add_column("Short Hash", justify="left", style="cyan", no_wrap=True)
            table.add_column("Entry Group", justify="left", style="cyan", no_wrap=True)
            table.add_column("Name", justify="left", style="magenta", no_wrap=True)
            if show_command:
                table.add_column("Command", justify="left", style="green", no_wrap=True)

            for entry in sorted(entries, key=lambda x: (x.group_name, x.name)):
                short_name = f"{entry.group_name}.{entry.name}"
                challenge_hash = challenge_hashes[group_name][challenge_name][:challenge_hash_length]
                entry_hash = entry_hashes[short_name][:entry_hash_length]
                if show_command:
                    table.add_row(
                        f"{challenge_hash}_{entry_hash}",
                        entry.group_name,
                        entry.name,
                        entry.command,
                    )
                else:
                    table.add_row(
                        f"{challenge_hash}_{entry_hash}",
                        entry.group_name,
                        entry.name,
                    )

            console.print(table)


cli.add_command(create_config)
cli.add_command(list_entries, name="list")

if __name__ == "__main__":
    cli()
