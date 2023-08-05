import numpy as np
import torch
from model import NeuralNet
from helpers import load_MNIST
import torch.nn as nn

config = {
    "job": {
        "status": False, # Flag indicating whether job is run succesfully
        "save_list": ["saved_results", "mnist", "cnn"],     # Root Save directory relative to project root directory  ["foo", "bar"] saves in <project_root>/foo/bar/
        "ID": None,                                                 # Chosen dynamically during job dispatching
        "job_dir": None,                                            # Chosen dynamically during job dispatching
        "params": None                                              # Gridsearched hyperparameters will be saved here.
    },

    # Most components follow roughly the same template:
    # A component name containing the function that should be executed
    # and the arguments of that function
    "data": {
        "loader": load_MNIST,
        "args": {
            "root_dir": 'files',  # Data directory relative to project root directory
            "batch_size_train": "gs",
            "batch_size_test": 4096
        }

    },
    "optimizer": {
        "algorithm": torch.optim.Adam,
        "args": {
            "lr": "gs",
        },
    },
    "criterion": {
        "algorithm": nn.CrossEntropyLoss,
        "args": {
        },
    },

    "model": {
        "algorithm": NeuralNet,
        "args": {
            "n_hidden": "gs",
            "actf": "gs"
        }
    },
    "trainer": {
        "N_epochs": 3,
        "drop_last": True,
    }
}

# Define hyperparameter space
lrs = [0.01, 0.02]
bs = [1024]
act_funcs = [nn.Tanh, nn.ReLU]
n_hiddens = [50]

gs_params_new = [
            {'n_hidden': n_hiddens, 'lr': lrs, 'batch_size_train': bs, 'actf': act_funcs}
            ]
