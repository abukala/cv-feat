import numpy as np

from skimage import color
from skimage.filters import gaussian
from scipy import misc
import logging


def rescale(image, max_value=1.0):
    if max_value == 255:
        return np.clip(image, 0, max_value).astype(np.uint8)
    else:
        return np.clip(image, 0, max_value).astype(np.float32)


def apply_gaussian_noise(image, std, mean=0.0, max_value=1.0):
    assert max_value in [1.0, 255]

    noisy = image + np.random.normal(mean, std, image.shape) * max_value
    noisy = rescale(noisy, max_value)

    return noisy


def apply_gaussian_blur(image, sigma, max_value=1.0, multichannel=False):
    return rescale(gaussian(image, sigma=sigma, multichannel=multichannel), max_value)


def apply_salt_and_pepper_noise(image, p, max_value=1.0):
    assert max_value in [1.0, 255]

    mask = np.random.random(image.shape)

    noisy = np.copy(image)
    noisy[mask < p / 2] = 0
    noisy[mask > (1 - p / 2)] = max_value

    return rescale(noisy, max_value)


def apply_quantization_noise(image, q, max_value=1.0):
    assert max_value in [1.0, 255]

    noisy = image + q * (np.random.random(image.shape) - 0.5) * max_value
    noisy = rescale(noisy, max_value)

    return noisy


def lower_resolution(image, scaling_factor, max_value=1.0):
    assert max_value in [1.0, 255]

    if max_value == 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        assert image.dtype == 'uint8'

    width = image.shape[0]
    height = image.shape[1]

    if len(image.shape) == 3:
        image_ycbcr = color.rgb2ycbcr(image)
        image_y = image_ycbcr[:, :, 0].astype(np.uint8)
    else:
        image_y = image.copy()

    downscaled = misc.imresize(image_y, 1 / float(scaling_factor), 'bicubic', mode='L')
    rescaled = misc.imresize(downscaled, (width, height), 'bicubic', mode='L').astype(np.float32)

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


def apply_occlusion(image, fraction):
    assert 0.0 <= fraction <= 1.0

    image_width, image_height = image.shape[:2]
    image_area = image_width * image_height
    occlusion_area = fraction * image_area
    occlusion_width = int(np.round(np.sqrt(occlusion_area)))

    if occlusion_width > image_width or occlusion_width > image_height:
        logging.warning('Occlusion larger than the image. Occlusion shape: (%d, %d), image shape: (%d, %d).' %
                     (occlusion_width, occlusion_width, image_width, image_height))

    occluded_image = image.copy()

    if occlusion_width < image_width:
        occlusion_start_x = np.random.randint(image_width - occlusion_width)
    else:
        occlusion_start_x = 0

    if occlusion_width < image_height:
        occlusion_start_y = np.random.randint(image_height - occlusion_width)
    else:
        occlusion_start_y = 0

    occluded_image[occlusion_start_x:(occlusion_start_x + occlusion_width),
    occlusion_start_y:(occlusion_start_y + occlusion_width)] = 0

    return occluded_image


def apply_noise(img, noise_type, noise_level):
    assert img.dtype in [np.float64, np.float32]
    assert img.max() <=1 and img.min() >= 0

    if noise_type == 'lres':
        noise_level = int(noise_level)
    else:
        noise_level = float(noise_level)

    if noise_type == 'gauss':
        noisy = apply_gaussian_noise(img, noise_level)
    elif noise_type == 'sp':
        noisy = apply_salt_and_pepper_noise(img, noise_level)
    elif noise_type == 'quantization':
        noisy = apply_quantization_noise(img, noise_level)
    elif noise_type == 'blur':
        noisy = apply_gaussian_blur(img, noise_level)
    elif noise_type == 'occlusion':
        noisy = apply_occlusion(img, noise_level)
    elif noise_type == 'lres':
        noisy = lower_resolution(img, noise_level)
    else:
        raise ValueError('Unknown noise_type: %s' % noise_type)

    assert noisy.dtype == img.dtype
    assert noisy.max() <= 1 and noisy.min() >= 0

    return noisy
