import numpy


class ImageDumper:
    def __init__(self, pose_keypoints, keypoints_file_name):
        self.pose_keypoints = pose_keypoints
        self.keypoints_file_name = keypoints_file_name

    def dump_image(self):
        first_person_keypoints = self.pose_keypoints.reshape((self.pose_keypoints.shape[1],
                                                              self.pose_keypoints.shape[2]))

        numpy.savetxt(f"image_keypoints/{self.keypoints_file_name}.csv", first_person_keypoints, delimiter=",")
