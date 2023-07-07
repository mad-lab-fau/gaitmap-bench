# FAU Woody Configuration

This directory contains scripts and configs to run gaitmap-bench on the FAU Woody cluster.
You will likely not have access to this cluster and can ignore these configs.

## Setup

1. From your local machine run the `force_copy_local.sh` script with your username as argument 
   to copy the entire repo to the cluster.
   All code will be copied to `/home/<username>/projects/gaitmap-bench`.
   
   ```bash
   ./force_copy_local.sh -u <username>
   ```
   
   Note that the script might throw multiple errors, if you run it the first time.
   Further, you might need to provide your SSH-key password multiple times.
   We recommend to setup ssh-agent properly to avoid this.
2. Ensure that all datasets you want to work with are available on the cluster.
   Copy them into `/home/<username>/datasets` and make sure that the names match the ones in the config file
   (`./config.json`).
   NOTE: AT THE MOMENT PATH ARE HARDCODED WITH A USERNAME! THIS NEEDS TO BE CHANGED!

   If you have the datasets locally in a similar folder structure, you can run:

   ```bash
   rsync -a . <username>@woody.nhr.fau.de:"/home/woody/iwso/<username>/datasets"
   ```
3. Setup Poetry on the cluster:

    ```bash
    module load python/3.9-anaconda # or whatever version you want to use
    curl -sSL https://install.python-poetry.org | python3 -
    ```
   You might need to update and source your `.bashrc` file afterwards.

4. Navigate into the gaitmap-bench project folder and run `poetry install`
5. Verify that everything worked by running `poetry run gaitmap-bench list`.
   This should print all available entries.

## Running Entries

To run an entry you need specify the Python interpreter, the config and an entrypoint script.
The entrypoint script handles the magic of spawning actual jobs on the cluster.
The following should work when you are in the project folder:

```bash
poetry run gaitmap-bench run <entry_id> -py $(which python3) -c .run_configs/fau_woody/config.json -e .run_configs/fau_woody/executor
```

Equivalently, you can use the `run-multi` command.

Remember to add the `--non-debug` flag, when you want to officially run the entry.
Otherwise, results will be stored with a debug flag and are not considered as official results for the website.

This will spawn one job per selected entry.
Note that the creation of the python env will still happen on the frontend node.

After the job is spawned, the script will terminate.
You can use `squeue` to check on the status of your jobs and monitor the stdout/stderr files in the `hpc_logs` folder.

Once all jobs are finished, you can download the results as explained below.

## Retrieving Results

Due to the way the system is setup, downloading the results is a little tricky.
While running a challenge entry, there is a chance that a new git commit is created in the repo.
The reason for that is we run poetry update before running the entry.
If the poetry.lock file changes, a new commit is created to ensure that we have a fixed repo state for each run.

This means when downloading the results, we also need to "download" the changed git repo.
To avoid issues with that, we treat the repo on the cluster as a remote and pull the changes locally using pull-rebase.
This will ensure that nothing breaks, even if you have local changes.

However, there are two big caveats:

1. You can not have uncommitted in your local repo.
   If you do, the pull-rebase will fail.
2. If you are on a different branch locally when you run the download script compared to when you submitted the jobs,
   things might break in unexpected ways.

To be safe, limit the amount of changes you make to the local repo while the jobs are running and commit everything 
(or at least stash it) before downloading the results.

To download the results, run the following command:

```bash
./result_sync.sh <username>
```

Alternatively, you can also run the `force_copy_local.sh` script again.
This will make sure that any local code changes you made will directly synced again with the cluster as well

```bash
./force_copy_local.sh -u <username>
`````

## Notes

- You might need to increase the wall time in `executor.sh` for long-running entries.
- You might want to increase the number of tasks (`ntasks`) in `executor.sh` for entries that use multiple processes.

   