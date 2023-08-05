import torch
import torchvision
import definitions as defs

def load_MNIST(batch_size_train, batch_size_test, root_dir):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(f'{defs.ROOT_DIR}/{root_dir}/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(f'{defs.ROOT_DIR}/{root_dir}/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True)
    return train_loader, test_loader


def sbatch_wrapper(config_id):
    # https://hpc.nmsu.edu/discovery/slurm/slurm-commands/
    # SBATCH --gres=gpu (first available GPU of any type)
    # SBATCH --gres=gpu:pascal:1 (one GPU, only Pascal type)
    # SBATCH --gres=gpu:2 (two GPUs for the same job)

    '''Wrapper to run script with slurm's sbatch. Hardcoded the script that it wraps.'''
    shell_str = f"""#!/bin/sh
    # You can control the resources and scheduling with '#SBATCH' settings
    # (see 'man sbatch' for more information on setting these parameters)

    # The default partition is the 'general' partition
    #SBATCH --partition=general

    # The default Quality of Service is the 'short' QoS (maximum run time: 4 hours)
    #SBATCH --qos=short

    # The default run (wall-clock) time is 1 minute
    #SBATCH --time=00:30:00

    # The default number of parallel tasks per job is 1
    #SBATCH --ntasks=1

    # Request 1 CPU per active thread of your program (assume 1 unless you specifically set this)
    # The default number of CPUs per task is 1 (note: CPUs are always allocated per 2)

    #SBATCH --cpus-per-task=2
    #SBATCH --gres=gpu

    # The default memory per node is 1024 megabytes (1GB) (for multiple tasks, specify --mem-per-cpu instead)
    #SBATCH --mem=4096

    #SBATCH --job-name={config_id}
    # Set mail type to 'END' to receive a mail when the job finishes
    # Do not enable mails when submitting large numbers (>20) of jobs at once
    ##SBATCH --mail-type=END

    # Your job commands go below here

    # Uncomment these lines when your job requires this software
    # module use /opt/insy/modulefiles
    # module load miniconda
    # module load cuda/11.1
    # module load cudnn/11.1-8.0.5.39
    conda activate heartpr

    # Complex or heavy commands should be started with 'srun' (see 'man srun' for more information)
    # For example: srun python my_program.py
    # Use this simple command to check that your sbatch settings are working (verify the resources allocated in the usage statistics)
    echo "Starting at $(date)"
    python run_NN.py --config_id={config_id}
    echo "Finished at $(date)"
    """.strip()
    return shell_str