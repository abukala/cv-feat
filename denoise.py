import numpy as np
import pathlib
from datasets import stl10, gtsrb, mnist, feret
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral
import json
import sys
import multiprocessing as mp


from noise import apply_gaussian_noise, apply_quantization_noise, apply_salt_and_pepper_noise, apply_gaussian_blur

RESULTS_PATH = pathlib.Path() / 'results' / 'baseline'

noise = {
    'gauss': apply_gaussian_noise,
    'quantization': apply_quantization_noise,
    'sp': apply_salt_and_pepper_noise,
    'blur': apply_gaussian_blur
}

noise_params = {
    'gauss': {
        'min': 0,
        'max': 0.25,
        'step': 0.025
    },
    'quantization': {
        'min': 0,
        'max': 0.5,
        'step': 0.05
    },
    'sp': {
        'min': 0,
        'max': 0.2,
        'step': 0.02
    },
    'blur': {
        'min': 0,
        'max': 5,
        'step': 0.5
    }
}


def psnr(x, y, maximum=255):
    return 20 * np.log10(maximum) - 10 * np.log10(np.mean(np.power(x - y, 2)))


def evaluate(noise_type, noise_level, images):
    methods = {
        'bm3d': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        'median': [3, 5, 7, 9, 11, 13],
        'bilateral': [(x, y) for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] for y in [3, 5, 7]]
    }

    result = {
        'input': [],
        'bm3d': {},
        'median': {},
        'bilateral': {}
    }

    for method in methods.keys():
        result[method] = {}

        for value in methods[method]:
            result[method][value] = []

    for img in images:
        if noise_level == 'random':
            if noise_type == 'random':
                n_params = noise_params[np.random.choice(list(noise_params.keys()))]
                noise_range = np.arange(n_params['min'], n_params['max'] + n_params['step'], n_params['step'])
                noisy = np.random.choice(list(noise.values()))(img, np.random.choice(noise_range))
            else:
                n_params = noise_params[noise_level]
                noise_range = np.arange(n_params['min'], n_params['max'] + n_params['step'], n_params['step'])
                noisy = noise[noise_type](img, np.random.choice(noise_range))
        else:
            noisy = noise[noise_type](img, noise_level)
        result['input'].append(psnr(img, noisy))

        for method in methods.keys():
            for value in methods[method]:
                if method == 'bm3d':
                    denoised = bm3d(noisy, value)
                elif method == 'median':
                    denoised = median(noisy, kernel_size=(value, value, 1))
                elif method == 'bilateral':
                    denoised = bilateral(noisy, sigma_range=value[0], sigma_spatial=value[1])
                else:
                    raise ValueError

                result[method][value].append(psnr(img, denoised))

    result['input'] = str(np.round(np.mean(result['input']), 2))

    for method in methods.keys():
        for value in methods[method]:
            result[method][value] = np.mean(result[method][value])

        result[method] = str(np.round(np.max(result[method].values()), 2))

    result['noise_type'] = noise_type
    result['noise_level'] = noise_level

    filename = '%s_%s.json' % (sys.argv[1], sys.argv[2])
    fp = RESULTS_PATH / filename
    with fp.open(mode='a') as output:
        json.dump(result, output)
        output.write('\n')


if __name__ == '__main__':
    assert len(sys.argv) == 3
    assert sys.argv[1] in ['gtsrb', 'mnist', 'stl10', 'feret']
    assert sys.argv[2] in ['gauss', 'sp', 'quantization', 'blur', 'random']
    dataset = eval(sys.argv[1])
    noise_type = sys.argv[2]
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    X, _ = dataset.load_training_data()
    if noise_type == 'random':
        mp.Process(target=evaluate, args=('random', 'random', X)).start()
    else:
        params = noise_params[noise_type]
        for noise_level in np.arange(params['min'] + params['step'], params['max'] + params['step'], params['step']):
            mp.Process(target=evaluate, args=(noise_type, noise_level, X)).start()
        mp.Process(target=evaluate, args=(noise_type, 'random', X)).start()

