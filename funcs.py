import copy
import os

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.model_selection import ParameterGrid
import definitions as defs
from utils import create_dir, load_pickle


class PopulateConfig:
    def __init__(self, config, params, old='gs'):
        self.config = config
        self.params = params
        self.old = old
        self.config_filled = copy.deepcopy(config)
        self.params_unused = copy.deepcopy(params)

    def dict_replace_value(self, config, old, new, key, params):
        '''modified from source: https://stackoverflow.com/questions/55704719/python-replace-values-in-nested-dictionary
           Iteratively traverses dict and modifies value of selected key.'''

        x = {}

        # Iterate over all config key-value pairs
        for k, v in config.items():

            # If value is a dict, go one level deeper in config and repeat
            if isinstance(v, dict):
                v, params = self.dict_replace_value(v, old, new, key, params)
                x[k] = v
            # If it's not a dict and the key of the config matches the
            # key of the hyperparameter we want to change
            else:
                x[k] = v
                if k == key:
                    # And the value is set as a gridsearchable hyperparameter
                    if v == old:
                        # Only then set the value
                        x[k] = new
                        params.pop(k)
                    else:
                        # If master config file has not set hyperparameter as gridsearchable,
                        # raise error
                        raise ValueError(f"Value is not gridsearchable according to config file. ({(k, v)})")
        return x, params

    def check_if_dict_filled(self, config_filled, val='gs'):
        '''Check if all gridsearchable hyperparameters in config file have been assigned a value'''
        x = {}

        # Iterate over all config key-value pairs
        for k, v in config_filled.items():

            # If value is a dict, go one level deeper in config and repeat
            if isinstance(v, dict):
                v = self.check_if_dict_filled(v)
                x[k] = v
            # If it's not a dict and gridsearchable hyperparameter's value is unassigned
            # raise error
            else:
                x[k] = v
                if v == val:
                    raise ValueError(f"Gridsearchable parameter {x} was not assigned a value in config.")
        return x

    def populate_config(self):

        # Find hyperparameter in config and replace value
        for k, v in self.params.items():
            self.config_filled, self.params_unused = self.dict_replace_value(self.config_filled, 'gs', v, k,
                                                                             self.params_unused)

        # Check if all hyperparameters have been assigned in config
        if self.params_unused:
            raise ValueError(
                f"Gridsearch object contains hyperparameters that were not set in config file:{self.params_unused}."
                " Probably the name is inconsistent with the key in the config file.")

        # Check for remaining hyperparameters in config that are remained unassigned
        self.check_if_dict_filled(self.config_filled, val='gs')

        return self.config_filled

def get_params(gs_params):
    gs_params_all = list(ParameterGrid(gs_params))
    gs_params_df = pd.DataFrame(gs_params_all)
    gs_params_df['status'] = False

    return gs_params_df

def gridsearch_union_grids(save_dir, gs_params_new):
    ''' Compares new gridsearch parameters with old ones that have already
     run (un)succesfully. Returns the union with index of the new parameters
     starting immediately after the last index of the old ones.'''
    create_dir(f"{save_dir}/jobs")  # Create job dirs if not exist

    if (job_dirs := os.listdir(f"{defs.ROOT_DIR}/{save_dir}/jobs")):
        # Fetch gridsearch params that have already been run (un)succesfuly.
        ids_old, status_old, gs_params_old = gridsearch_aggregate_params(job_dirs, save_dir)

        # Index of new jobs begin at one after the last index of the old jobs
        # gs_params_new.index += max(ids_old) + 1
        gs_params_old = pd.DataFrame(gs_params_old, index=ids_old)
        gs_params_old['status'] = status_old
        gs_params_old = gs_params_old.sort_index()

        # Create placeholder value -1 for the indices of the new jobs, so we can recognize them easy later
        gs_params_new.index = np.ones(len(gs_params_new)) * -1
        gs_params = pd.concat((gs_params_old, gs_params_new))

        # TODO: Pandas cannot drop duplicates of lists. Identify columns which contain lists, convert to str
        # and convert back to list after dropping duplicates
        # Drop duplicates only based on parameter values without status column.

        # Remove jobs already run
        subs = list(gs_params.columns.values)
        subs.remove('status')
        gs_params = gs_params.drop_duplicates(subset=subs)

        # TODO:Transform string lists back to list
        # gs_params.param = gs_params.param.apply(lambda x: eval(x))
        gs_params = gs_params.astype(
            gs_params_new.dtypes.to_dict())  # Make sure dtypes are not changed to unexpected ones
        # After removing duplicates assign an ID to the new jobs (starting at the largest ID in the old jobs + 1)
        # Note this manipulates the dataframe by reference
        tot_ids = gs_params.index.to_numpy()
        tot_ids[tot_ids == -1] = tot_ids.max() + np.arange(1, len(tot_ids[tot_ids == -1]) + 1)

        gs_params.index = gs_params.index.astype('int')
    else:
        # There are no old jobs, so just run these jobs as is.
        gs_params = gs_params_new

    return gs_params

def gridsearch_aggregate_params(dirs, save_dir):
    ''' Aggregate gridsearch parameters used for all jobs'''
    # gridsearch params that have already been run (un)succesfuly.
    gs_params = []
    status = []
    ids = []
    for job in dirs:
        try:
            config_job = load_pickle(f"{save_dir}/jobs/{job}", "config")

            status.append(config_job['job']['status'])
            gs_params.append(config_job['job']['params'])

            # Fetch all job ids. Extract number part of each job directory name and convert to int.
            ids.append(int(job[3:]))
        except FileNotFoundError:
            print(f"Config file not found for {job}. Skipping..")

    return ids, status, gs_params