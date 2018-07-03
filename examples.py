from skimage import data
from skimage.io import imsave
from skimage.color import rgb2gray
from skimage.transform import resize
from noise import apply_noise
import numpy as np
import pathlib

EXAMPLES_PATH = pathlib.Path() / 'examples'

distortions = {
    'gauss': [0.1, 0.2],
    'sp': [0.05, 0.15],
    'quantization': [0.2, 0.4],
    'blur': [2, 4],
    'occlusion': [0.2, 0.4]
}

if __name__ == '__main__':
    EXAMPLES_PATH.mkdir(parents=True, exist_ok=True)
    clean = rgb2gray(resize(data.astronaut(), (128, 128))).astype(np.float32)
    fp = EXAMPLES_PATH / 'example_clean.png'
    imsave(fp, clean)
    for dist, dist_levels in distortions.items():
        for dist_level in dist_levels:
            distorted = apply_noise(clean, dist, dist_level)
            fp = EXAMPLES_PATH / ('example_%s_%s.png' % (dist, dist_level))
            imsave(fp, distorted)