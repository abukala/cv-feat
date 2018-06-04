from databases import RESULTS_PATH
from noise import noise_params
import pathlib
import numpy as np
RESULTS_PATH = RESULTS_PATH / 'baseline'
AUX_DENOISE_PATH = pathlib.Path() / 'aux_denoise.sh'

for dataset in ['gtsrb', 'stl10', 'feret', 'mnist']:
    for noise_type in ['gauss', 'sp', 'quantization', 'random']:
        filename = '%s_%s.json' % (dataset, noise_type)
        fp = RESULTS_PATH / filename
        if noise_type == 'random':
            noise_range = []
        else:
            noise_range = [nr for nr in np.arange(noise_params[noise_type]['min'] + noise_params[noise_type]['step'],
                                              noise_params[noise_type]['max'], noise_params[noise_type]['step'])]
        noise_range.append('random')
        if fp.exists():
            with fp.open() as file:
                data = [eval(line) for line in file.readlines()]

            for noise_level in noise_range:
                found = False
                for line in data:
                    try:
                        if round(noise_level, 3) == round(line['noise_level'], 3):
                            found = True
                            break
                    except TypeError:
                        if noise_level == line['noise_level']:
                            found = True
                            break
                if not found:
                    with AUX_DENOISE_PATH.open(mode='a') as f:
                        print("sbatch slurm.sh denoise.py %s %s %s" % (dataset, noise_type, noise_level), file=f)
        else:
            for noise_level in noise_range:
                with AUX_DENOISE_PATH.open(mode='a') as f:
                    print("sbatch slurm.sh denoise.py %s %s %s" % (dataset, noise_type, noise_level), file=f)