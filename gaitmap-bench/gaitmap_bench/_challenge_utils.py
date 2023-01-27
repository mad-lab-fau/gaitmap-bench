import os
from typing import Dict, Any

import pandas as pd
from gaitmap_challenges import BaseChallenge


def save(metadata: Dict[str, Any], challenge: BaseChallenge) -> None:
    """Save the results of an entry in the correct folder structure."""
    # TODO: Create folder structure based on metadata
    # Find the file, the function was called from
    # Check git history and save commit hash with results
    # If "drity" git repo, save it as unofficial run
    # Save version of challenge
    # Collect info about host PC
    print(metadata)
    results = pd.DataFrame(challenge.cv_results_)
    with pd.option_context("display.max_columns", 120):
        print(results.filter(like="test_per_sample"))
