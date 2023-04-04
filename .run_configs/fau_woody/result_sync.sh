#!/bin/bash

# Get the directory path of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Define the relative path to the directory you want to sync
REL_PATH="../../results/"

# Get the username from the command-line arguments
USERNAME="$1"

# Remote path to the gaitmap-bench directory
REMOTE_PATH="~/projects/gaitmap-bench"

# Use rsync to sync the directory to the remote machine
rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/results" "$PROJECT_ROOT"
# Note, we also need to sync the git directory and the entries folder, as running an entry can create a commit for an
# updated poetry.lock file
# WARNING: If you made any git changes locally, while the remote machine was running, they will be overwritten!
rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/.git" "$PROJECT_ROOT"
rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/entries" "$PROJECT_ROOT"

