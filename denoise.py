import numpy as np
import pathlib
from datasets import stl10, gtsrb, mnist, feret
from skimage.measure import compare_psnr, compare_ssim
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral
from skimage.restoration import unsupervised_wiener as wiener
import json
import sys
import multiprocessing as mp
import operator
from tqdm import tqdm
from noise import apply_noise
from skimage import img_as_float

RESULTS_PATH = pathlib.Path() / 'results' / 'baseline'

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


def denoise(img, method, value):
    assert method in ['bm3d', 'median', 'bilateral', 'wiener']
    assert img.dtype == np.float32
    assert img.max() <= 1 and img.min() >= 0

    if method == 'bm3d':
        denoised = bm3d(img, value)
    elif method == 'median':
        denoised = median(img, kernel_size=(value, value))
    elif method == 'bilateral':
        denoised = bilateral(img, sigma_color=value[0], sigma_spatial=value[1], multichannel=False)
    elif method == 'wiener':
        psf = np.ones((value[0], value[0])) / value[1]
        denoised, _ = wiener(img, psf)
    else:
        raise ValueError('Unknown method: %s' % method)

    denoised = img_as_float(denoised).astype(np.float32)

    assert denoised.dtype == img.dtype
    assert denoised.max() <= 1 and denoised.min() >= 0, (denoised.max(), denoised.min())

    return denoised


def evaluate(noise_type, noise_level, images):
    methods = {
        'bm3d': [0.05, 0.1, 0.2, 0.4, 0.5],
        'median': [3, 5, 7, 9, 11, 13],
        'bilateral': [(x, y) for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] for y in [3, 5, 7]],
        'wiener': [(x, y) for x in range(1, 6) for y in range(1, 26)]
    }

    result = {
        'input': []
    }
    if noise_type in ['gauss', 'sp', 'quantization']:
        tested_methods = ['bm3d', 'median', 'bilateral']
        measure = compare_psnr
    elif noise_type == 'blur':
        tested_methods = ['wiener']
        measure = compare_ssim
    else:
        raise ValueError('No denoising methods known for %s' % noise_type)

    for method in tested_methods:
        result[method] = {}

        for value in methods[method]:
            result[method][value] = []

    assert len(images[0].shape) == 2

    for i, img in tqdm(enumerate(images), total=len(images)):
        if noise_level == 'random':
            if noise_type == 'random':
                n_type = np.random.choice(list(noise_params.keys()))
                n_params = noise_params[n_type]
                noise_range = np.arange(n_params['min']+n_params['step'], n_params['max'] + n_params['step'], n_params['step'])
                noisy = apply_noise(img, n_type, np.random.choice(noise_range))
            else:
                n_params = noise_params[noise_type]
                noise_range = np.arange(n_params['min']+n_params['step'], n_params['max'] + n_params['step'], n_params['step'])
                noisy = apply_noise(img, noise_type, np.random.choice(noise_range))
        else:
            noisy = apply_noise(img, noise_type, noise_level)

        result['input'].append(measure(img, noisy))

        for method in tested_methods:
            for value in methods[method]:
                denoised = denoise(noisy, method, value)
                result[method][value].append(measure(img, denoised))

    result['input'] = str(np.round(np.mean(result['input']), 2))

    for method in tested_methods:
        for value in methods[method]:
            result[method][value] = np.mean(result[method][value])

        result[method] = max(result[method].items(), key=operator.itemgetter(1))
        result[method] = result[method][0], round(result[method][1], 2)

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
    print('Searching for optimal denoise params: %r' % sys.argv[1:])
    dataset = eval(sys.argv[1])
    noise_type = sys.argv[2]
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    X, _ = dataset.load_training_data()

    if sys.argv[1] == 'feret':
        np.random.seed(0)
        np.random.shuffle(X)
        X = X[:1000]

    assert X[0].dtype == np.float32
    assert X[0].max() <= 1 and X[0].min() >= 0

    if noise_type == 'random':
        mp.Process(target=evaluate, args=('random', 'random', X)).start()
    else:
        params = noise_params[noise_type]
        for noise_level in np.arange(params['min'] + params['step'], params['max'] + params['step'], params['step']):
            mp.Process(target=evaluate, args=(noise_type, noise_level, X)).start()
        mp.Process(target=evaluate, args=(noise_type, 'random', X)).start()

