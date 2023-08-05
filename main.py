import os
import sys
import subprocess
from helpers import sbatch_wrapper
from run_NN import run_NN
from funcs import gridsearch_union_grids, PopulateConfig
from utils import save_pickle
from funcs import get_params
import time
import definitions
################ Choose a config file here ################
from configs import mnist_config as conf
###########################################################
platform ='local'
gs_params_new_df = get_params(conf.gs_params_new)
config = conf.config

# Define root dir of gridsearch
save_dir = "/".join(config['job']['save_list'])

# %%
gs_params = gridsearch_union_grids(save_dir, gs_params_new_df)

# Save gridsearch parameters and master configuration file in experiment directory
save_pickle(save_dir, "gs_params", gs_params)
save_pickle(save_dir, "master_config", config)

# Shuffle to do random search
# gs_params = gs_params.sample(frac=1)
# Don't bother with jobs that did not converge anymore..
gs_params = gs_params.loc[gs_params['status'] > -1]

for i, (ind, params) in enumerate(gs_params.iterrows()):
    print(f'{params} ({i + 1}/{len(gs_params)})')
    params_dict = params.drop('status').to_dict()

    # Define path to configuration (job)
    job_dir = f"{save_dir}/jobs/job{ind}"

    if not params['status']:
        # create_dir(job_dir)
        # Add results directory for this config to config dict

        # Fill master config file with this configuration
        pop_config = PopulateConfig(config, params_dict)
        config_filled = pop_config.populate_config()

        config_filled['job']['ID'] = ind
        config_filled['job']['job_dir'] = job_dir
        config_filled['job']['params'] = params_dict

        # If running on SLURM, save the config with job ID as name
        # so we can load it in the run script later
        if platform == 'sbatch':

            sbatch_conf = sbatch_wrapper(config_filled['job']['ID'])
            # Create bash file with sbatch settings
            with open(f"{definitions.ROOT_DIR}/tmp/job{config_filled['job']['ID']}.sh", 'w') as file:
                file.write(sbatch_conf)

            # Save populated config file to use later in run file
            save_pickle(f"tmp", f"config_job{config_filled['job']['ID']}", config_filled)
            time.sleep(0.1)
            subprocess.Popen(['sbatch', f"{definitions.ROOT_DIR}/tmp/job{config_filled['job']['ID']}.sh"], shell=True)
        else:
            status = run_NN(config_filled)
            if status:
                txt = f"Job {i + 1} of {len(gs_params)} completed"

    else:
        print(f'job {ind} already run.')
        print("")