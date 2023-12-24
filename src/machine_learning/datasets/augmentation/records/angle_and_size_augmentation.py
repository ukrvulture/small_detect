#!/usr/bin/python3

import collections
import json
import os
import re


RecordFileDesc = collections.namedtuple('RecordFile', ['suffix'])


FILE_NAME_FORBIDDEN_SYMBOLS = ''.join([
    '~', r'\)', r'\(', r'\\', r'\!', r'\*', r'\<', r'\>',
    r'\:', r'\;' r'\,', r'\?', r'\"', r'\*', r'\|', r'\/'])
FILE_NAME_RE_PATTERN = '[^' + FILE_NAME_FORBIDDEN_SYMBOLS + ']+'


class AngledResizedSrcInTargetFileDesc:
    """Descriptor of resized and rotated object augmented in target image."""

    AUGMENTED_FULL = RecordFileDesc('full')
    SRC_MASK_FULL = RecordFileDesc('full.mask')

    TARGET_TILE = RecordFileDesc('target')
    AUGMENTED_TILE = RecordFileDesc(suffix=None)
    AUGMENTED_SRC = RecordFileDesc('augsrc')
    SRC_MASK = RecordFileDesc('mask')

    ALL_RECORD_FILES = {TARGET_TILE, AUGMENTED_TILE, AUGMENTED_SRC, SRC_MASK}

    FILE_NAME_RE = re.compile('^' + ''.join([
        r'(?P<augmented_file_prefix>' + FILE_NAME_RE_PATTERN + '~' + FILE_NAME_RE_PATTERN + ')',
        r'\.a(?P<angle>\d+)', r'\.w(?P<width>\d+)',
        r'(\.(?P<suffix>\w+))?',
        r'\.(?P<ext>\w+)']) + '$')

    def __init__(self, combined_augmented_file_prefix=''):
        self.combined_augmented_file_prefix = combined_augmented_file_prefix
        self.angle_in_degrees = 0
        self.scaled_width_in_pixels = 0

    def __eq__(self, other):
        return (
            self.angle_in_degrees == other.angle_in_degrees and
            self.scaled_width_in_pixels == other.scaled_width_in_pixels and
            self.combined_augmented_file_prefix == other.combined_augmented_file_prefix)

    def __hash__(self):
        return hash((self.combined_augmented_file_prefix,
            self.angle_in_degrees, self.scaled_width_in_pixels))

    def __repr__(self):
        return "{}.a{:03}.w{:03}".format(self.combined_augmented_file_prefix,
            self.angle_in_degrees, self.scaled_width_in_pixels)

    @classmethod
    def all_file_descriptors(cls):
        return cls.ALL_RECORD_FILES

    @classmethod
    def from_src_and_target_file(cls, src_img_file, target_image_file):
        return cls(combined_augmented_file_prefix=cls.combine_augmented_file_names(
            target_image_file, src_img_file))

    @staticmethod
    def combine_augmented_file_names(src_img_file, target_image_file):
        return f'{src_img_file.stem}~{target_image_file.stem}'

    def combine_record_file_name(self, file_desc, ext):
        if file_desc.suffix:
            return "{}.a{:03}.w{:03}.{}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels, file_desc.suffix, ext)
        else:
            return "{}.a{:03}.w{:03}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels, ext)

    @classmethod
    def parse_from_file_path(cls, file_path):
        file_name = os.path.basename(file_path) if isinstance(file_path, str) else file_path.name
        name_match = cls.FILE_NAME_RE.match(file_name)
        if name_match:
            parsed_desc = AngledResizedSrcInTargetFileDesc(
                combined_augmented_file_prefix=name_match.group('augmented_file_prefix'))
            parsed_desc.angle_in_degrees = int(name_match.group('angle'))
            parsed_desc.scaled_width_in_pixels = int(name_match.group('width'))
            return parsed_desc, RecordFileDesc(name_match.group('suffix'))
        else:
            return None, None


class AngledResizedSrcInTargetFileGenDesc(AngledResizedSrcInTargetFileDesc):
    """Descriptor of resized and rotated object augmented in target image with generation info."""

    def __init__(self, combined_augmented_file_prefix=''):
        super().__init__(combined_augmented_file_prefix)
        self.tile_top_left_row = 0
        self.tile_top_left_col = 0
        self.tile_width = 0
        self.tile_height = 0

    def __repr__(self):
        return json.dumps(vars(self), indent=2)
