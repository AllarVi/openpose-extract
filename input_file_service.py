import logging
from glob import glob

from path import Path


class InputFileService:

    @staticmethod
    def get_input_files(input_files_path):
        input_files = [Path(f).abspath() for f in glob(input_files_path + '/*')]
        logging.info(f"Found {len(input_files)} files in source directory")

        return input_files
