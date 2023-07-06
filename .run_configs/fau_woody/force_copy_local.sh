#!/bin/bash

# Get the directory path of the script
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Remote path to the gaitmap-bench directory
REMOTE_PATH="~/projects/"

# Get commandline arguments
USERNAME=""
DELETE_REMOTE_RESULTS=false

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -u|--username) USERNAME="$2"; shift ;;
        -d|--delete-results) DELETE_REMOTE_RESULTS=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Sync the directory to the remote machine using the original script if the SYNC_ONLY flag is not set
if [ "$DELETE_REMOTE_RESULTS" = false ] ; then
  echo "Copying results from remote to avoid overwriting them"
  /bin/bash "$SCRIPT_DIR/result_sync.sh" "$USERNAME"
  echo "Copying .git and entries folder from remote to avoid overwriting them"
  # Note: We need this, as running an entry can create a commit for an updated poetry.lock file
  #rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/gaitmap-bench/.git" "$PROJECT_ROOT"
  rsync -a --ignore-existing "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH/gaitmap-bench/entries" "$PROJECT_ROOT"
fi


# Use rsync to sync the directory to the remote machine
echo "Copying everything from $PROJECT_ROOT to $REMOTE_PATH on woody"
rsync -a --filter=':- dir-merge,-n /.gitignore' --delete "$PROJECT_ROOT" "$USERNAME"@woody.nhr.fau.de:"$REMOTE_PATH"
