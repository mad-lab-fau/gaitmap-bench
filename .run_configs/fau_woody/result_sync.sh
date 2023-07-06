#!/bin/bash

# Get the directory path of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Get the username from the command-line arguments
USERNAME="$1"

# Remote path to the gaitmap-bench directory
REMOTE_PATH="~/projects/gaitmap-bench"

# Note, we need to sync the git directory and the entries folder, as running an entry can create a commit for an
# updated poetry.lock file
# To do this safely without deleting any commits locally, we actually use git and not rsync for this.
git remote add _woody "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH"
git pull _woody "$(git branch --show-current)" --rebase || exit 1
git remote remove _woody

# Use rsync to sync the results directory from the remote machine
rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/results" "$PROJECT_ROOT"


