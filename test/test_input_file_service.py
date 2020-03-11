from unittest import TestCase

from input_file_service import InputFileService


class TestInputFileService(TestCase):

    def test_get_input_files(self):
        input_files = InputFileService.get_input_files(".")

        self.assertTrue(len(input_files) > 1)
