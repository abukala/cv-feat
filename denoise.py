import numpy as np
import pathlib
from datasets import stl10, gtsrb, mnist, feret
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral
import time

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
        noisy = noise[noise_type](img, noise_level)
        result['input'].append(psnr(clean, noisy))

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

                result[method][value].append(psnr(clean, denoised))

    result['input'] = str(np.round(np.mean(result['input']), 2))

    for method in methods.keys():
        for value in methods[method]:
            result[method][value] = np.mean(result[method][value])

        result[method] = str(np.round(np.max(result[method].values()), 2))

    return result


if __name__ == '__main__':
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for dataset in [gtsrb, mnist, stl10, feret]:
        X, _ = dataset.load_training_data()
        clean = X[:10]
        ts = time.time()
        for noise_type in noise:
            params = noise_params[noise_type]
            for noise_level in np.arange(params['start'], params['stop'] + params['step'], params['step']):
                results[noise] = evaluate(noise_type, noise_level, clean)
        evaluation_time = round(time.time() - ts, 1)
        print('Evaluation time: %s seconds' % evaluation_time)
        print('Approx time for dataset: %s minutes' % ((len(X) * evaluation_time) / 600))
