#!/usr/bin/python3

# Methods to combine image rotation, resizing and cropping.

from src.img_processing.base import image_pool

import numpy as np


def rotate_resize_crop_rgba_img(
    img_rgba, angle_in_degrees, scaled_width_in_pixels,
    alpha_channel_threshold):
    """Rotates using the given angle and resizes proportionally with a given width.

    Args:
      img_rgba: Source image RGBa-array.
      angle_in_degrees: Rotation angle in degrees [0..360].
      scaled_width_in_pixels: Target width in pixels.
      alpha_channel_threshold: Threshold to filter out transparent pixels [0..255].

    Returns:
      RGBa-array with rotated and resized image
      which all the rows and columns have at least one non-transparent pixel.
    """
    with image_pool.ImagePool() as img_pool:
        # img_rgba = scipy.ndimage.rotate(img_rgba, angle=angle_in_degrees, reshape=True)
        rotated_img = img_pool.imagick_from_rgba(img_rgba)
        rotated_img.rotate(-angle_in_degrees)
        img_rgba = np.array(rotated_img)

        # Clear boundary artifacts.
        transparent_pixels = img_rgba[:, :, 3] < alpha_channel_threshold

        visible_min_r, visible_max_r = 0, img_rgba.shape[0] - 1
        visible_min_c, visible_max_c = 0, img_rgba.shape[1] - 1

        while np.all(transparent_pixels[visible_min_r, :]):
            visible_min_r += 1
        while np.all(transparent_pixels[visible_max_r, :]):
            visible_max_r -= 1
        while np.all(transparent_pixels[:, visible_min_c]):
            visible_min_c += 1
        while np.all(transparent_pixels[:, visible_max_c]):
            visible_max_c -= 1

        rotated_and_resized_rgba = np.copy(img_rgba[
            visible_min_r:visible_max_r + 1, visible_min_c:visible_max_c + 1,:])

        scaled_height_in_pixels = int(rotated_and_resized_rgba.shape[0] *
            scaled_width_in_pixels / rotated_and_resized_rgba.shape[1])

        # rotated_and_resized_rgba = skimage.transform.resize(
        #    rotated_and_resized_rgba, (scaled_height_in_pixels, scaled_width_in_pixels),
        #    order=1, mode='constant', anti_aliasing=False,
        #    preserve_range=True).astype(np.uint8)
        rotated_and_resized_img = img_pool.imagick_from_rgba(rotated_and_resized_rgba)
        rotated_and_resized_img.resize(
            width=scaled_width_in_pixels, height=scaled_height_in_pixels, filter='lanczos')
        rotated_and_resized_rgba = np.array(rotated_and_resized_img)

        rotated_and_resized_rgba[
           rotated_and_resized_rgba[:, :, 3] < alpha_channel_threshold] = [0, 0, 0, 0]
        rotated_and_resized_rgba[
           rotated_and_resized_rgba[:, :, 3] >= alpha_channel_threshold, 3] = 255

        return rotated_and_resized_rgba
