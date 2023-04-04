#!/bin/bash

# Get the directory path of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

# Define the relative path to the directory you want to sync
REL_PATH="../../results/"

# Get the username from the command-line arguments
USERNAME="$1"

# Remote path to the gaitmap-bench directory
REMOTE_PATH="~/projects/gaitmap-bench"

# Use rsync to sync the directory to the remote machine
rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/results/" "$SCRIPT_DIR/$REL_PATH"

