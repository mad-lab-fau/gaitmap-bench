import hashlib
from typing import Set, Sequence, Dict, List

import click
import pandas as pd
from rich.console import Console
from rich.table import Table

from gaitmap_bench import create_config_template
from gaitmap_bench._config import DEFAULT_CONFIG_FILE, DEFAULT_ENTRIES_DIR
from gaitmap_bench._utils import find_all_entries, Entry


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
        def filter_group(entry: Entry) -> bool:
            if entry.challenge_group_name in group:
                return True
            if entry.challenge_group_name + "." + entry.challenge_name in group:
                return True
            return False

        all_entries = [e for e in all_entries if filter_group(e)]

    min_hash_length = _determine_shortest_required_length(
        [e.hash for e in all_entries], [3, 6, 9]
    )

    entries_df = pd.DataFrame(all_entries)

    if len(entries_df) == 0:
        console.print("No entries found.")
        return

    for group_id, entries in entries_df.groupby(["challenge_group_name", "challenge_name"]):
        group_name, challenge_name = group_id
        table = Table(title=f"{group_name}.{challenge_name}  ({len(entries)} entries)")
        table.add_column("ID", justify="left", style="cyan", no_wrap=True)
        table.add_column("Entry Group", justify="left", style="cyan", no_wrap=True)
        table.add_column("Name", justify="left", style="magenta", no_wrap=True)
        table.add_column("Full Name", justify="left", style="magenta", no_wrap=True)
        if show_command:
            table.add_column("Command", justify="left", style="green", no_wrap=True)

        for entry in entries.sort_values(["group_name", "name"]).itertuples(name="Entry"):
            entry: Entry
            columns = [entry.hash[:min_hash_length], entry.group_name, entry.name, entry.run_name]
            if show_command:
                columns.append(entry.command)
            table.add_row(*columns)

        console.print(table)


@cli.command()
@click.option("--id", "-i", "entry_id", type=str, help="The ID of the entry to run.")
def run_challenge(entry_id):
    """Run a challenge."""



cli.add_command(create_config)
cli.add_command(list_entries, name="list")
cli.add_command(run_challenge, name="run")

if __name__ == "__main__":
    cli()
