#!/usr/bin/python3

import wand.image


class ImagePool:
    """Managed pool to control memory allocated for images."""

    def __init__(self):
        self.created_imagick_images = []

    def imagick_from_rgba(self, rgba):
        imagick_img = wand.image.Image.from_array(rgba, channel_map='RGBA')
        self.created_imagick_images.append(imagick_img)
        return imagick_img

    def imagick_from_imagick(self, imagick_img):
        cloned_img = imagick_img.clone()
        self.created_imagick_images.append(cloned_img)
        return cloned_img

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for imagick_img in self.created_imagick_images:
            imagick_img.close()
