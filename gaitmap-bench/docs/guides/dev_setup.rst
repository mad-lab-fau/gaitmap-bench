Both gaitmap-bench and gaitmap-challenges have their own venv (that is just how poetry works).

However, we basically only use the venv for gaitmap-bench, as gaitmap-challenges is a dependency of gaitmap-bench.
I.e. the gaitmap-bench venv contains all the same packages as the gaitmap-challenges venv, plus some more.

This means we should point pycharm to the gaitmap-bench venv, and not the gaitmap-challenges venv.

During dev, we likely want to use gaitmap-challenges as a local dependency, so we can make changes to it and see the effects in gaitmap-bench.
To do this, we need to add gaitmap-challenges as a local dependency in gaitmap-bench's pyproject.toml file.
Note, we need to update this again to point to the package on pypi before publishing.