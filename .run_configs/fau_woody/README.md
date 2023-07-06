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

## Retrieving Results

   