#!/usr/bin/python3

import os


class RawImage:
    """Image pixels and meta-data representation."""

    def __init__(self, path='', rgba=None):
        self.file_name = os.path.basename(path) if path else 'undefined'
        self.path = ""
        self.rgba = rgba

    def __repr__(self):
        return self.file_name
