import argparse
import logging
import random
from argparse import Namespace
from pathlib import Path
from time import gmtime

import numpy as np
import torch



def setup_output_directory(output_dir, create=True):
    """
    Sets up the output directory. This means either creating one, or
    verifying that the provided directory exists. Output directories
    are created using the run and experiment ids.
    Args:
        output_dir (str): The target output directory
        run_uuid : The current hash of the current run.
        experiment_id: The id of the current experiment
        create (bool): Boolean indicating whether to create a new folder.
    """

    if create:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    elif not Path(output_dir).exists():
        raise FileNotFoundError(
            'Output directory does not exist: {}'.format(output_dir)
        )

    return output_dir

def configure_seed(seed):
    """
    Configure the random seed for all relevant packages.
    These include: random, numpy, torch and torch.cuda
    Args:
        seed (int): the random seed to be set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

