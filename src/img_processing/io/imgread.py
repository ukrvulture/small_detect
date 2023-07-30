#!/usr/bin/python3

# Helpers for image reading.

import logging
import os
import pathlib

import skimage.io


def load_images_from_dir(dir_path, ignore_non_images=False, recursive=False):
    """Loads images from disk skipping auxiliary files.

    Args:
      dir_path: Folder path.
      ignore_non_images: Just skip, if there is an auxiliary files in the folder.
      recursive: Flag to look also in sub-folders.

    Returns:
      Generator yielding file path and image pixel array.
    """
    sub_paths = []
    for file_name in os.listdir(dir_path):
        file_path = pathlib.Path(os.path.join(dir_path, file_name))
        if file_path.is_dir():
            sub_paths.append(file_path)
            continue

        try:
            yield file_path, skimage.io.imread(str(file_path))
        except ValueError:
            if not ignore_non_images:
                logging.warning(f"{file_path} is not a file")
            continue

    if recursive and sub_paths:
        for dir_path in sub_paths:
            yield from load_images_from_dir(dir_path, ignore_non_images, recursive)
