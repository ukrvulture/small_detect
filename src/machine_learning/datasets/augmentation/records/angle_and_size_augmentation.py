#!/usr/bin/python3

import collections
import json
import os
import re


RecordFileDesc = collections.namedtuple('RecordFileDesc', ['suffix'])


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
    TARGET_MASK = RecordFileDesc('mask')
    TARGET_SCALED_MASK = RecordFileDesc('scaled_mask')
    TARGET_SCALED_MASK_OF_MASK = RecordFileDesc('scaled_mask_mask')

    ALL_RECORD_FILES = {
        TARGET_TILE, AUGMENTED_TILE, AUGMENTED_SRC, TARGET_MASK, TARGET_SCALED_MASK, TARGET_SCALED_MASK_OF_MASK}

    FILE_NAME_RE = re.compile('^' + ''.join([
        r'(?P<augmented_file_prefix>' + FILE_NAME_RE_PATTERN + '~' + FILE_NAME_RE_PATTERN,
        r'\.a(?P<angle>\d+)', r'\.w(?P<width>\d+)', r'\.t(?P<top>\d+)', r'\.l(?P<left>\d+)', ')',
        r'(\.(?P<suffix>\w+))?',
        r'\.(?P<ext>\w+)']) + '$')

    def __init__(self, file_path, combined_augmented_file_prefix='', target_file_desc=None):
        self.file_path = file_path
        self.target_file_desc = target_file_desc
        self.combined_augmented_file_prefix = combined_augmented_file_prefix

        self.tile_top_left_row = 0
        self.tile_top_left_col = 0
        self.angle_in_degrees = 0
        self.scaled_width_in_pixels = 0

    def __eq__(self, other):
        return (
            # self.angle_in_degrees == other.angle_in_degrees and
            # self.scaled_width_in_pixels == other.scaled_width_in_pixels and
            # self.tile_top_left_row == other.tile_top_left_row and
            # self.tile_top_left_col == other.tile_top_left_col and
            self.combined_augmented_file_prefix == other.combined_augmented_file_prefix and
            self.target_file_desc == other.target_file_desc)

    def __hash__(self):
        return hash((
            # self.angle_in_degrees, self.scaled_width_in_pixels,
            # self.tile_top_left_row, self.tile_top_left_col,
            self.combined_augmented_file_prefix, self.target_file_desc ))

    def __repr__(self):
        return "{}.a{}.w{}.t{}.l{}".format(self.combined_augmented_file_prefix,
            self.angle_in_degrees, self.scaled_width_in_pixels,
            self.tile_top_left_row, self.tile_top_left_col)

    @classmethod
    def all_file_descriptors(cls):
        return cls.ALL_RECORD_FILES

    @classmethod
    def from_src_and_target_file(cls, src_img_file, target_image_file):
        return cls(file_path=None, combined_augmented_file_prefix=cls.combine_augmented_file_names(
            target_image_file, src_img_file))

    @staticmethod
    def combine_augmented_file_names(src_img_file, target_image_file):
        return f'{src_img_file.stem}~{target_image_file.stem}'

    def combine_record_file_name(self, file_desc, ext):
        if file_desc.suffix:
            return "{}.a{}.w{}.t{}.l{}.{}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels,
                self.tile_top_left_row, self.tile_top_left_col, file_desc.suffix, ext)
        else:
            return "{}.a{}.w{}.t{}.l{}.{}".format(self.combined_augmented_file_prefix,
                self.angle_in_degrees, self.scaled_width_in_pixels,
                self.tile_top_left_row, self.tile_top_left_col, ext)

    @classmethod
    def parse_from_file_path(cls, file_path):
        file_name = os.path.basename(file_path) if isinstance(file_path, str) else file_path.name
        name_match = cls.FILE_NAME_RE.match(file_name)
        if name_match:
            parsed_desc = AngledResizedSrcInTargetFileDesc(
                file_path, combined_augmented_file_prefix=name_match.group('augmented_file_prefix'),
                target_file_desc=RecordFileDesc(name_match.group('suffix')))
            parsed_desc.angle_in_degrees = int(name_match.group('angle'))
            parsed_desc.scaled_width_in_pixels = int(name_match.group('width'))
            return parsed_desc
        else:
            return None


class AngledResizedSrcInTargetFileGenDesc(AngledResizedSrcInTargetFileDesc):
    """Descriptor of resized and rotated object augmented in target image with generation info."""

    def __init__(self, file_path, combined_augmented_file_prefix=''):
        super().__init__(file_path, combined_augmented_file_prefix)
        self.tile_width = 0
        self.tile_height = 0

    def __repr__(self):
        return json.dumps(vars(self), indent=2)
