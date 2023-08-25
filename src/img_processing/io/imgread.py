#!/usr/bin/python3

# Helpers for image reading.

from src.img_processing.base import image

import logging
import os
import pathlib

import numpy as np
import skimage.io


def load_images_from_dir(dir_path, ignore_non_images=False, recursive=False):
    """Loads images from disk skipping auxiliary files.

    Args:
      dir_path: Folder path.
      ignore_non_images: Just skip, if there is an auxiliary files in the folder.
      recursive: Flag to look also in sub-folders.

    Returns:
      Generator yielding RawImage object.
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
            continue

    if recursive and sub_paths:
        for dir_path in sub_paths:
            yield from load_images_from_dir(dir_path, ignore_non_images, recursive)
