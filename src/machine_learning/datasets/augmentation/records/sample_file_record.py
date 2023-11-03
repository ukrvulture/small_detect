#!/usr/bin/python3


class SampleFileRecord:
    """All artifacts of ore training image sample."""

    def __init__(self, record_file_desc):
        self.record_file_desc = record_file_desc
        self.record_files_by_desc = {}

    def __eq__(self, other):
        return self.record_file_desc == other.record_file_desc

    def __hash__(self):
        return hash(self.record_file_desc)

    def __repr__(self):
        return '{}[{}]'.format(repr(self.record_file_desc), ','.join(sorted(
            (file_desc.suffix if file_desc.suffix else '-')
            for file_desc in self.record_files_by_desc.keys())))

    @property
    def desc(self):
        return self.record_file_desc

    def all_files(self):
        return (len(self.record_file_desc.all_file_descriptors().difference(
            self.record_files_by_desc.keys())) == 0)

    def get_saved_file_path(self, dir_path, file_desc, ext):
        saved_file_path = dir_path.joinpath(self.desc.combine_record_file_name(file_desc, ext))
        self.record_files_by_desc[file_desc] = saved_file_path
        return saved_file_path

    def add_file(self, file_desc, file_path):
        self.record_files_by_desc[file_desc] = file_path


class SampleFileRecordDist(dict):
    """Dictionary wrapper to collect training image samples."""

    def add(self, file_record_desc, file_desc, file_path):
        if file_record_desc not in self:
            self[file_record_desc] = SampleFileRecord(file_record_desc)
        self[file_record_desc].add_file(file_desc, file_path)
