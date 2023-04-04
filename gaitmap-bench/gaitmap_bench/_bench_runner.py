import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import click
import pandas as pd
from gaitmap_challenges.config import _CONFIG_ENV_VAR, _DEBUG_ENV_VAR
from rich.console import Console
from rich.table import Table

from gaitmap_bench import create_config_template
from gaitmap_bench._config import DEFAULT_CONFIG_FILE, DEFAULT_ENTRIES_DIR, MAIN_REPO_ROOT
from gaitmap_bench._utils import Entry, find_all_entries


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
        if len({h[:length] for h in hashes}) == len(hashes):
            return length

    return len(hashes[0])


@click.group()
def cli():
    r"""Run and configure the benchmark suite.

    It can be used for the following tasks:

    \b
    - Create a template config for your local configuration.
    - List all benchmarks entries that are registered in the `entries` folder.
    - Run individual benchmarks.
    - (Future) scaffold a new entry to a benchmark.
    """


@cli.command()
@click.argument("path", type=click.Path(exists=False), default=DEFAULT_CONFIG_FILE)
def create_config(path):
    """Create a template config file for the benchmarking suite."""
    create_config_template(path)


@cli.command()
@click.option(
    "--path", "-p", type=click.Path(exists=False), default=DEFAULT_ENTRIES_DIR, help="The path to the entries folder."
)
@click.option(
    "--group",
    "-g",
    multiple=True,
    type=str,
    default=None,
    help="Only list entries of the given group. (Can be specified multiple times)",
)
@click.option("--show-command", "-c", is_flag=True, help="Show the command that is used to run the entry.")
@click.option("--show-base-folder", "-f", is_flag=True, help="Show the base folder of the entry.")
def list_entries(path, group, show_command, show_base_folder):
    """List all entries that are registered in the `entries` folder."""
    path = Path(path)
    all_entries = find_all_entries(path)

    min_hash_length = _determine_shortest_required_length([e.hash for e in all_entries], [3, 6, 9])

    console = Console()

    if group:

        def filter_group(entry: Entry) -> bool:
            if entry.challenge_group_name in group:
                return True
            if entry.challenge_group_name + "." + entry.challenge_name in group:
                return True
            return False

        all_entries = [e for e in all_entries if filter_group(e)]

    entries_df = pd.DataFrame(all_entries)

    if len(entries_df) == 0:
        console.print("No entries found.")
        return

    display_path = path.relative_to(MAIN_REPO_ROOT) if path == DEFAULT_ENTRIES_DIR else path

    for group_id, entries in entries_df.groupby(["challenge_group_name", "challenge_name"]):
        group_name, challenge_name = group_id
        table = Table(title=f"{group_name}.{challenge_name}  ({len(entries)} entries)")
        table.add_column("ID", justify="left", style="cyan", no_wrap=True)
        table.add_column("Entry Group", justify="left", style="cyan", no_wrap=True)
        table.add_column("Name", justify="left", style="magenta", no_wrap=True)
        table.add_column("Full Name", justify="left", style="magenta", no_wrap=True)
        if show_base_folder:
            table.add_column("Base Folder", justify="left", style="green", no_wrap=True)
        if show_command:
            table.add_column("Command", justify="left", style="blue", no_wrap=True)
        entry: Entry
        for entry in entries.sort_values(["group_name", "name"]).itertuples(name="Entry"):
            columns = [entry.hash[:min_hash_length], entry.group_name, entry.name, entry.run_name]
            if show_base_folder:
                columns.append(str(display_path / entry.base_folder))
            if show_command:
                columns.append(entry.command_template.format(command=entry.command))
            table.add_row(*columns)

        console.print(table)


@cli.command()
@click.option(
    "--path", "-p", type=click.Path(exists=False), default=DEFAULT_ENTRIES_DIR, help="The path to the entries folder."
)
@click.option(
    "--python-path",
    "-py",
    type=click.Path(exists=True),
    required=True,
    help="The path to the python executable to use.",
)
@click.option("--id", "-i", "entry_id", type=str, help="The ID of the entry to run.")
@click.option(
    "--config-path",
    "-c",
    type=click.Path(exists=True),
    help="The path to the config file. "
    "Will be resolved to an absolut path before passing it as an ENV variable to the child process.",
)
@click.option(
    "--non-debug",
    "-nd",
    is_flag=True,
    help="If set, the benchmark will be run in non-debug mode (i.e. debug=False). "
    "This should be used for official results and will enable additional checks to ensure reproducibility.",
)
@click.option(
    "--executor",
    "-e",
    type=str,
    default="",
    help="An executor command that will be called with the `command` as the first and the name of entry as "
         "second argument. "
         "This should be a path to a script relative to the working directory you started the command in. "
         "Note, that it requires the right permissions to be executable. "
)
def run_challenge(entry_id, path, python_path, config_path, non_debug, executor):
    """Run a challenge."""
    path = Path(path)
    all_entries = pd.DataFrame(find_all_entries(path))
    # Find the entry whichs hash starts with the given id
    entry = all_entries[all_entries.hash.str.startswith(entry_id)]
    if len(entry) == 0:
        raise ValueError(
            f"Found no entry with the given ID: {entry_id}\n"
            "Rerun the `list` command to see all entries and double-check the ID."
        )
    if len(entry) > 1:
        raise ValueError(
            f"Found multiple entries with the given ID: {entry_id}\n"
            "Rerun the `list` command to see all entries and double-check the ID."
        )
    entry = entry.iloc[0]

    # We make the following run variables available to the setup and command templates
    run_variables = {
        "command": entry.command,
        "python_path": python_path,
    }

    working_path = path / entry.base_folder
    command = entry.command_template.format(**run_variables)
    if executor:
        resolved_executor = Path(executor).resolve()
        command_with_executor = f'{resolved_executor} "{command}" "{entry.run_name}"'
    else:
        command_with_executor = command
    setup = entry.setup
    if not isinstance(setup, list):
        setup = [setup]
    setup_commands = [s.format(**run_variables) for s in setup]

    console = Console()
    console.print(f"Running entry: [bold blue]{entry.run_name}[/bold blue] ({entry.hash})")
    console.rule("[bold red]Run Plan[/bold red]")
    console.print("Executing the following commands:")
    for s in setup_commands:
        console.print(f"\t{s}")
    console.print(f"\t{command_with_executor}")
    console.print(f"In the following folder:\n\t{working_path}")
    console.rule("[bold red]Setup[/bold red]")

    new_env = os.environ.copy()
    # We unset VIRTUAL_ENV to make sure that the setup commands are executed in the correct environment
    new_env.pop("VIRTUAL_ENV", None)
    # We set the path to the config file as an environment variable
    new_env[_CONFIG_ENV_VAR] = str(Path(config_path).resolve())
    # We set the debug flag as an environment variable
    new_env[_DEBUG_ENV_VAR] = str(not non_debug)

    try:
        for s in setup_commands:
            console.log(f"Executing: {s}")
            subprocess.run(
                s, cwd=working_path, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, env=new_env
            )

    except subprocess.CalledProcessError as e:
        console.print_exception()
        console.print(f"Setup failed with error code {e.returncode}. See error above.")
        return
    console.rule("[bold red]Running[/bold red]")
    try:
        console.log(f"Executing: {command}")
        subprocess.run(
            command_with_executor, cwd=working_path, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr, env=new_env
        )
    except subprocess.CalledProcessError as e:
        console.print_exception()
        console.print(f"Executing the Command failed with error code {e.returncode}. See error above.")
        return


cli.add_command(create_config)
cli.add_command(list_entries, name="list")
cli.add_command(run_challenge, name="run")

if __name__ == "__main__":
    cli()
