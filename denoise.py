import numpy as np
import pathlib
from datasets import stl10, gtsrb, mnist, feret
from skimage.measure import compare_psnr
from pybm3d.bm3d import bm3d
from scipy.signal import medfilt as median
from skimage.restoration import denoise_bilateral as bilateral
from skimage.color import gray2rgb, rgb2gray
import json
import operator
import sys
import multiprocessing as mp
from tqdm import tqdm
from noise import apply_noise, rescale

RESULTS_PATH = pathlib.Path() / 'results' / 'baseline'


def denoise(img, method, value):
    assert img.dtype == np.float32
    assert img.max() <= 1 and img.min() >= 0

    if method == 'bm3d':
        denoised = bm3d(gray2rgb(img), value)
        denoised = rgb2gray(rescale(denoised))
    elif method == 'median':
        denoised = median(img, kernel_size=(value, value))
    elif method == 'bilateral':
        denoised = bilateral(img, sigma_color=value[0], sigma_spatial=value[1], multichannel=False)
    else:
        raise ValueError('Unknown method: %s' % method)

    denoised = rescale(denoised)

    assert denoised.dtype == img.dtype

    return denoised


def run(method, X_clean, X_noisy):
    methods = {
        'bm3d': [0.05, 0.1, 0.2, 0.4, 0.5],
        'median': [3, 5, 7, 9, 11, 13],
        'bilateral': [(x, y) for x in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5] for y in [3, 5, 7]],
    }

    assert len(X_clean[0].shape) == 2 and len(X_noisy[0].shape) == 2
    psnr = {}

    for value in methods[method]:
        psnr[value] = []

    for clean, noisy in tqdm(zip(X_clean, X_noisy), total=len(X_clean)):
        for value in methods[method]:
            denoised = denoise(noisy, method, value)
            while np.count_nonzero(np.isnan(denoised)) != 0:
                denoised = denoise(apply_noise(clean, result['noise_type'], result['noise_level']), method, value)
            psnr[value].append(compare_psnr(clean, denoised))

    for value in psnr:
        psnr[value] = str(np.round(np.mean(psnr[value]), 2))

    best_result = max(psnr.items(), key=operator.itemgetter(1))
    result[method] = best_result
    if 'bm3d' in result and 'bilateral' in result and 'median' in result:
        print('saving...')
        filename = '%s_%s.json' % (sys.argv[1], sys.argv[2])
        fp = RESULTS_PATH / filename
        with fp.open(mode='a') as output:
            json.dump(dict(result), output)
            output.write('\n')


if __name__ == '__main__':
    assert len(sys.argv) >= 3
    assert sys.argv[1] in ['gtsrb', 'mnist', 'stl10', 'feret']
    print('Searching for optimal denoise params: %r' % sys.argv[1:])
    dataset = eval(sys.argv[1])
    noise_type = sys.argv[2]
    noise_level = sys.argv[3]
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    X, _ = dataset.load_training_data()

    if sys.argv[1] == 'feret':
        np.random.seed(0)
        np.random.shuffle(X)
        X = X[:1000]

    assert X[0].dtype == np.float32
    assert X[0].max() <= 1 and X[0].min() >= 0

    if noise_level != 'random':
        noise_level = eval(noise_level)

    Xn = [apply_noise(img, noise_type, noise_level) for img in X]
    manager = mp.Manager()

    result = manager.dict({
        'noise_type': noise_type,
        'noise_level': noise_level,
        'input': str(np.round(np.mean([compare_psnr(clean, noisy) for clean, noisy in zip(X, Xn)]), 2))
    })
    proc = [mp.Process(target=run, args=(denoise_method, X, Xn)) for denoise_method in ['bm3d', 'bilateral', 'median']]
    for p in proc:
        p.start()
    for p in proc:
        p.join()
