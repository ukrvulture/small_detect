#!/usr/bin/python3

import os, sys
SCRIPT_DIRS = os.path.dirname(os.path.abspath(__file__)).split(os.sep)
sys.path.append(os.path.join(os.sep, *SCRIPT_DIRS[:SCRIPT_DIRS.index('src')]))

from src.machine_learning.datasets.augmentation.records.angle_and_size_augmentation import AngledResizedSrcInTargetFileDesc
from src.machine_learning.datasets.augmentation.records.sample_file_record import SampleFileRecord

import argparse
import collections
import glob
import logging
import pathlib


def parse_args(argv):
    # Parse input arguments.
    parser = argparse.ArgumentParser(description='Fit Sources into Targets')
    parser.add_argument('-d', '--dataset_dir', dest='root_dataset_dir',
                        help='Root of all the dataset files.', required=True)

    return parser.parse_args(argv[1:])


def main(argv):
    parsed_args = parse_args(argv)

    root_dataset_dir = parsed_args.root_dataset_dir

    aug_samples = collections.defaultdict(set)

    for globPath in glob.iglob(os.path.join(root_dataset_dir, '**/**'), recursive=True):
        checked_path = pathlib.Path(globPath)
        if checked_path.is_file():
            aug_file_desc = AngledResizedSrcInTargetFileDesc.parse_from_file_path(checked_path)
            if aug_file_desc:
                aug_samples[aug_file_desc.combined_augmented_file_prefix].add(aug_file_desc)

    # TODO (odruzh): validate samples
    for sample_key, sample_files in aug_samples.items():
        sample_files = set([sample_file.target_file_desc for sample_file in sample_files])
        if sample_files != set(AngledResizedSrcInTargetFileDesc.ALL_RECORD_FILES):
            logging.error("%s has missing sample files.", sample_key)

    logging.info("%d total samples.", len(aug_samples))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    sys.exit(main(sys.argv))
