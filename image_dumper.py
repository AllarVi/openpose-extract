import logging

import numpy


class ImageDumper:
    def __init__(self, pose_keypoints, keypoints_file_name):
        self.pose_keypoints = pose_keypoints
        self.keypoints_file_name = keypoints_file_name

    def dump_image(self):
        logging.info("Dumping frame to file...")

        (person_indices, x, y) = self.pose_keypoints.shape

        for person_idx in range(person_indices):
            numpy.savetxt(f"image_keypoints/{self.keypoints_file_name}-{person_idx}.csv",
                          self.pose_keypoints[person_idx],
                          delimiter=",",
                          header="x,y,c",
                          comments='')
