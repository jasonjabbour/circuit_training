import os

from absl import app
import subprocess
import itertools
import pathlib


def create_exp_name(hparams):
    return '_'.join([f'{k}_{v:03d}' if isinstance(v, int) else f'{k}_{v}' for k, v in hparams.items()])


def main(_):
    hparam_sweeps = {
            'cores': [160, ],
            'gpus': [1, 2, ],
            'collect_jobs': [4, 6],
            'seed': [111, 222, 333]
            }

    circuit_training_dir = pathlib.Path(__file__).parent.parent.parent.resolve()

    hparam_tuple = tuple(hparam_sweeps.items())
    hparam_tuple = sorted(hparam_tuple, key=lambda x: x[0])
    for hparam in itertools.product(*(hparam_sweeps[k] for k, v in hparam_tuple)):
        hparam_dict = dict(zip((k for k, v in hparam_tuple), hparam))

        exp_name = create_exp_name(hparam_dict)
        log_path = os.path.join(circuit_training_dir, 'logs', exp_name)
        out_file = os.path.join(log_path, '%j.out')
        err_file = os.path.join(log_path, '%j.err')

        pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
        cores = hparam_dict.get('cores', -1)
        collect_jobs = hparam_dict.get('collect_jobs', -1)
        gpus = hparam_dict.get('gpus', 1)
        seed = hparam_dict.get('seed', 47)
        command = f'''sbatch <<EOT
#!/bin/bash
#SBATCH --job-name=ariane_{exp_name}
#SBATCH -n {cores} # Number of cores requested
#SBATCH -t 2-00:00:00         # Runtime in D-HH:MM:SS, minimum of 10 minutes
#SBATCH -p seas_dgx1_priority # Partition to submit to
#SBATCH --mem=40000 # Memory per core in MB
#SBATCH --open-mode=append # Append when writing files
#SBATCH --gres=gpu:{gpus} # Number of GPUs to use
#SBATCH -D {circuit_training_dir}  # Change the working directory to the main circuit training directory
#SBATCH -o {out_file}  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e {err_file}  # File to which STDERR will be written, %j inserts jobid
export SINGULARITYENV_NUM_CT_COLLECT_JOBS={collect_jobs}
export SINGULARITYENV_ROOT_DIR={log_path}
export SINGULARITYENV_GLOBAL_SEED={seed}
export SINGULARITYENV_XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda/bin
export SINGULARITYENV_TF_FORCE_GPU_ALLOW_GROWTH=true
singularity exec --nv {os.path.join(circuit_training_dir, 'circuit_training.sif')} \
bash -x {os.path.join(circuit_training_dir, 'launch', 'ariane_risc_v_cpu', 'launch.sh')} 
exit 0
EOT
'''
        print(f'Running experiment {exp_name}')
        subprocess.run(command, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    app.run(main)
