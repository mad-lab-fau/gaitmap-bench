Development Setup
=================

.. warning:: This is intended for developers that want to add a new challenge or make other changes to either the
   gaitmap-bench or gaitmap-challenges code bases.
   If you want to add a new entry to an existing challenge, check out this `guide <new_entry>`_.

The project is split into multiple python projects that all manage their own dependencies using poetry.
This can become messy fast, when you want to change

To simplify things, we use a single venv for all the projects.
Unfortunately, this is something that is not really supported by poetry.

To workaround this limitation, we have a top-level pyproject.toml file that includes all the other projects as local
dependencies editable dependencies.
Further, this toplevel project contains all the dev dependencies (formatter, linter, etc.) and their config.

For this setup to work, we need to consider a couple of things:

First Setup
-----------

1. Install poetry (ideally >1.5.0)
2. Decide how you want to mange your Python versions (pyenv, conda, etc.)
3. Use a Python 3.8 or larger version as your default python version
4. Run `poetry install` in the root directory of this project. This will create a new venv. Depending on your global
   poetry config, this venv will either be in the root directory of this project, or in your poetry cache directory.
   Check with `poetry env list` where the venv is located.
5. Use this venv for your project in your IDE (e.g. pycharm). If you need to run commands from the command line, you
   can activate the venv with `poetry shell` or `poetry run <command>`.

Adding Dependencies
-------------------

As we only have a single top-level venv, we want all dependencies to be installed there.
However, we also want to mange the dependencies of each project separately in their own pyproject.toml file.

To achieve this, the process of adding new dependencies is a little bit more complicated than usual.

Let's start with the simple part: Adding a new dev dependency to the top-level project.
The top-level project just behaves like any other poetry project, so we can just add the dependency to the
pyproject.toml or use `poetry add <package> --group dev`.

Adding a new dependency to one of the sub-projects is a little bit more complicated.
For this, navigate to the sub-project and then use `poetry add <package> --lock` to add the dependency.
The `--lock` is crucial, as this will tell poetry that to not install the dependency, but just add it to the lock file.
Now we need to install the package.
For this, we go back to the top-level project and run `poetry update <sub-package>`.

Similarly, if you want ot update a dependency (or update the lock file, after changing the `pyproject.toml`), you need
to run `poetry update --lock` in the sub-project (remember, the `--lock` is crucial) and then
`poetry update <sub-package>` in the top-level project.

Developing Entries in Parallel
------------------------------

Often when making changes to the challenges, you will want to test them using one of the entries, or you might even
need to make changes to the entry itself.
Normally, each entry has their own isolated venv to ensure reproducibility.
This is annoying when working in a single IDE, as you need to switch between the venvs all the time.

To make this process a little easier, you can install the entries dependencies in the top-level venv.
At the moment, we only have one entry group (`gaitmap_algos`).
Hence, for convenience, we have set this one up as an optional dependency in the toplevel project.

You can install the top-level project and all dependencies of the `gaitmap_algos` group with
`poetry install --with gaitmap_algos`.

If we have other entry groups in the future, we might consider adding them as optional dependencies as well.
However, this would require all entries to be compatible with each other (as poetry resolves them all at once).
This means this might not be possible for all entries.
But, we will cross this bridge when we get there.
For future reference, there is an open issue for `poetry` that proposes exclusive groups (see
`here <https://github.com/python-poetry/poetry/issues/1168>`_).
If this is implemented, this could be a solution for this problem.
