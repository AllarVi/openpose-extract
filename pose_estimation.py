import cv2

from image_dumper import ImageDumper


def get_params():
    params = dict()
    params["logging_level"] = 3
    params["output_resolution"] = "-1x-1"
    params["net_resolution"] = "-1x368"
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    params["scale_gap"] = 0.3
    params["scale_number"] = 1
    params["render_threshold"] = 0.05
    params["num_gpu_start"] = 0  # If GPU version is built, and multiple GPUs are available, set the ID here
    params["disable_blending"] = False
    params["model_folder"] = "/Users/allarviinamae/EduWorkspace/openpose/models"
    return params


def main():
    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
        # OpenPose/python module from there. This will install OpenPose and the python library at your desired
        # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
        # '/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        print(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e

    params = get_params()

    # Constructing OpenPose object allocates GPU memory
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()

    # Opening OpenCV stream
    video_to_process = 'rasmus-flack-backflip.avi'
    video_capture = cv2.VideoCapture(f'/Users/allarviinamae/EduWorkspace/openpose-sample-videos/{video_to_process}')

    frame_index = 0

    while True:

        retval, image = video_capture.read()
        frame_index = frame_index + 1

        if frame_index == 10:
            break

        if not retval:
            break

        # Output keypoints and the image with the human skeleton blended on it
        datum = op.Datum()
        datum.cvInputData = image

        op_wrapper.emplaceAndPop([datum])
        print(f"Pose estimated for frame={frame_index}")

        # Print the human pose keypoints, i.e., a [#people x #keypoints x 3]-dimensional numpy object with the
        # keypoints of all the people on that image
        # print(datum.poseKeypoints)
        image_dumper = ImageDumper(datum.poseKeypoints, f"{video_to_process}-{frame_index}")
        image_dumper.dump_image()

        # Display the stream
        cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    video_capture.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
