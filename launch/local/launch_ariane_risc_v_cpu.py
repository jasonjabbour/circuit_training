import os

from absl import app
import subprocess
import itertools
import pathlib


def create_exp_name(hparams):
    return '_'.join([f'{k}_{v:03d}' if isinstance(v, int) else f'{k}_{v}' for k, v in hparams.items()])


def main(_):
    hparam_sweeps = {
            'seed': [333]
            }

    host_ct_dir = pathlib.Path(__file__).parent.parent.parent.resolve()

    hparam_tuple = tuple(hparam_sweeps.items())
    hparam_tuple = sorted(hparam_tuple, key=lambda x: x[0])
    for hparam in itertools.product(*(hparam_sweeps[k] for k, v in hparam_tuple)):
        hparam_dict = dict(zip((k for k, v in hparam_tuple), hparam))
        exp_name = create_exp_name(hparam_dict)

        host_log_path = os.path.join(host_ct_dir, 'logs', exp_name)
        vm_log_path = os.path.join('/circuit_training', 'logs', exp_name)

        pathlib.Path(host_log_path).mkdir(parents=True, exist_ok=True)

        collect_jobs = hparam_dict.get('collect_jobs', 2)
        seed = hparam_dict.get('seed', 47)

        command = f'''bash <<EOF
docker run --env ROOT_DIR={vm_log_path} \
    --env GLOBAL_SEED={seed} \
    --env NUM_CT_COLLECT_JOBS={collect_jobs} \
    --env TF_FORCE_GPU_ALLOW_GROWTH=true \
    --rm \
    -v {host_ct_dir}:/circuit_training \
    --workdir=/circuit_training \
    --gpus all \
    circuit_training:core \
    {os.path.join('/circuit_training', 'launch', 'ariane_risc_v_cpu', 'launch.sh')}
EOF
'''

        print(command)
        print(f'Running experiment {exp_name}')
        subprocess.run(command, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    app.run(main)
