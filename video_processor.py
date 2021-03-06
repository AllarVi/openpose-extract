import logging
import time

import cv2
from openpose import pyopenpose as op

from image_dumper import ImageDumper


class VideoProcessor:

    def __init__(self, op_wrapper, show_video) -> None:
        super().__init__()
        self.op_wrapper = op_wrapper
        self.show_video = show_video

    def process(self, video_to_process):
        logging.info(f"Starting processing of {video_to_process.split('/')[-1]}...")
        video_capture = cv2.VideoCapture(video_to_process)

        frame_index = 0

        while True:
            start = VideoProcessor.current_time_sec()

            retval, frame = video_capture.read()
            frame_index = frame_index + 1

            if not retval:
                logging.info(f"Return value={retval}. Ending video processing")
                break

            # Output keypoints and the image with the human skeleton blended on it
            datum = op.Datum()
            datum.cvInputData = frame

            self.op_wrapper.emplaceAndPop([datum])
            logging.info(f"Pose estimated for frame = {frame_index}")

            # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the
            # keypoints of all the people on that image
            if len(datum.poseKeypoints.shape) == 0:
                print("WARN! Pose keypoints not found! Skipping to next frame")
                continue

            video_to_process_filename = video_to_process.split("/")[-1]  # extract filename
            image_dumper = ImageDumper(datum.poseKeypoints, video_to_process_filename)
            image_dumper.dump_image(frame_index)

            # Display the stream
            if self.show_video:
                cv2.imshow("OpenPose - Python API", datum.cvOutputData)
                key = cv2.waitKey(1)

                if key == ord('q'):
                    break

            stop = round(VideoProcessor.current_time_sec() - start, 2)
            logging.info(f"Frame processing time {stop} seconds")

        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def current_time_sec():
        return round(time.time(), 2)
