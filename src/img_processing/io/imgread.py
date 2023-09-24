#!/usr/bin/python3

# Helpers for image reading.

from src.img_processing.base import image

import logging
import os
import re

import imghdr
import numpy as np
import skimage.io


def list_image_file_from_dir(
    dir_path, recursive=False, ignore_non_images=False, ignored_file_regexp=None):
    """Lists absolute paths to image files.

    Args:
      dir_path: Root folder path.
      recursive: Flag to look also in sub-folders.
      ignore_non_images: Just skip, if there is an auxiliary files in the folder.
      ignored_file_regexp: Python compiled regexp pattern or sting of ignored file names.

    Returns:
      Generator yielding ImageFile objects.
    """
    if ignored_file_regexp and isinstance(ignored_file_regexp, str):
        skip_file_regexp = re.compile(ignored_file_regexp)

    sub_paths = []
    for file_name in sorted(os.listdir(str(dir_path))):
        file_path = dir_path / file_name
        if file_path.is_dir():
            sub_paths.append(file_path)
            continue

        if ignored_file_regexp and ignored_file_regexp.match(file_name):
            continue

        if imghdr.what(str(file_path)):
            yield image.ImageFile(file_path.absolute())
        elif not ignore_non_images:
            logging.warning(f"{file_path} is not an image.")

    if recursive and sub_paths:
        for dir_path in sub_paths:
            yield from list_image_file_from_dir(dir_path, ignore_non_images, recursive)


def load_images_from_dir(dir_path, recursive=False, ignore_non_images=False):
    """Loads images from disk skipping auxiliary files.

    Args:
      dir_path: Folder path.
      recursive: Flag to look also in sub-folders.
      ignore_non_images: Just skip, if there is an auxiliary files in the folder.

    Returns:
      Generator yielding RawImage objects.
    """
    sub_paths = []
    for file_name in sorted(os.listdir(str(dir_path))):
        file_path = dir_path / file_name
        if file_path.is_dir():
            sub_paths.append(file_path)
            continue

        try:
            img_rgba = skimage.io.imread(str(file_path))
            if img_rgba.shape[-1] == 3:
                img_rgba = np.insert(img_rgba, 3, 255, axis=2)
            yield image.RawImage(path=file_path, rgba=img_rgba)
        except ValueError:
            if not ignore_non_images:
                logging.warning(f"{file_path} is not an image.")

    if recursive and sub_paths:
        for dir_path in sub_paths:
            yield from load_images_from_dir(dir_path, ignore_non_images, recursive)
