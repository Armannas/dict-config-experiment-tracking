import os
import pandas as pd
import definitions as defs
from configs import mnist_config as conf
from utils import load_pickle

# Analyze results of the gridsearch
config = conf.config
proj_dir = "/".join(config['job']['save_list'])
jobs_root_dir = f"{proj_dir}/jobs"

job_dirs = os.listdir(f"{defs.ROOT_DIR}/{jobs_root_dir}")

# Extract results and parameters in list of dicts
# Then we can make a dataframe out of it later
res_dicts = []
for job_dir in job_dirs:
    job_abs_dir = f"{jobs_root_dir}/{job_dir}"
    job_conf = load_pickle(f"{job_abs_dir}", "config")
    perf_res = load_pickle(f"{job_abs_dir}/results", "perf_res")

    # Only consider performance of last epoch
    perf_res_last_epoch = {k:v[-1] for k,v in perf_res.items()}

    # Extract parameters used in this configuration
    res_dict = job_conf['job']['params']

    # Merge with performance results
    res_dict = {**res_dict, **perf_res_last_epoch}

    res_dicts.append(res_dict)

res_df = pd.DataFrame.from_dict(res_dicts)
print(res_df)

# Get job id with best test accuracy
best_id = res_df.test_accs.idxmax()

job_abs_dir_best = f"{jobs_root_dir}/job{best_id}"
perf_res_best = load_pickle(f"{job_abs_dir_best}/results", "perf_res")

