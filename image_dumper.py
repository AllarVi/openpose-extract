import logging
import os
from os import path

import numpy


class ImageDumper:
    def __init__(self, pose_keypoints, video_to_process_filename):
        self.pose_keypoints = pose_keypoints
        self.video_to_process_filename = video_to_process_filename

    def dump_image(self, frame_index):
        logging.info("Dumping frame to file...")

        output_dir = f"output/{self.video_to_process_filename.split('.')[0]}"  # extract filename without extension

        if not path.exists(output_dir):
            logging.info(f"Creating output dir={output_dir}")
            os.mkdir(output_dir)

        (person_indices, x, y) = self.pose_keypoints.shape

        for person_idx in range(person_indices):
            numpy.savetxt(
                f"{output_dir}/{self.video_to_process_filename}-{frame_index}-{person_idx}.csv",
                self.pose_keypoints[person_idx],
                delimiter=",",
                header="x,y,c",
                comments='')
