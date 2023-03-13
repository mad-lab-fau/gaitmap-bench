"""

"""
from pathlib import Path

import click

from gaitmap_bench import create_config_template

MAIN_REPO_ROOT = Path(__file__).parent.parent.parent


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
@click.argument("path", type=click.Path(exists=False), default=MAIN_REPO_ROOT / ".dev_config.json")
def create_config(path):
    """Create a template config file for the benchmarking suite."""
    create_config_template(path)


cli.add_command(create_config)

if __name__ == "__main__":
    cli()
