import numpy as np

from skimage import color
from scipy import misc


def rescale(image, max_value=255):
    assert max_value in [1.0, 255]

    return np.clip(image, 0, max_value)


def apply_gaussian_noise(image, std, mean=0.0, max_value=255):
    assert max_value in [1.0, 255]

    noisy = image + np.random.normal(mean, std, image.shape) * max_value
    noisy = rescale(noisy, max_value)

    return noisy


def apply_salt_and_pepper_noise(image, p, max_value=255):
    assert max_value in [1.0, 255]

    mask = np.random.random(image.shape)

    noisy = np.copy(image)
    noisy[mask < p / 2] = 0
    noisy[mask > (1 - p / 2)] = max_value

    return noisy


def apply_quantization_noise(image, q, max_value=255):
    assert max_value in [1.0, 255]

    noisy = image + q * (np.random.random(image.shape) - 0.5) * max_value
    noisy = rescale(noisy, max_value)

    return noisy


def lower_resolution(image, scaling_factor, max_value=255):
    assert max_value in [1.0, 255]

    if max_value == 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        assert image.dtype == 'uint8'

    width = image.shape[0]
    height = image.shape[1]
    width = width - width % scaling_factor
    height = height - height % scaling_factor
    image = image[:width, :height]

    if len(image.shape) == 3:
        image_ycbcr = color.rgb2ycbcr(image)
        image_y = image_ycbcr[:, :, 0].astype(np.uint8)
    else:
        image_y = image.copy()

    downscaled = misc.imresize(image_y, 1 / float(scaling_factor), 'bicubic', mode='L')
    rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='L').astype(np.float32)

    if len(image.shape) == 3:
        low_res_image = image_ycbcr
        low_res_image[:, :, 0] = rescaled
        low_res_image = color.ycbcr2rgb(low_res_image)
        low_res_image = (np.clip(low_res_image, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        low_res_image = rescaled.astype(np.uint8)

    if max_value == 1.0:
        return low_res_image / 255
    else:
        return low_res_image