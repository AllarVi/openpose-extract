import getopt
import logging
import sys

from input_file_service import InputFileService
from logging_config import LoggingConfig
from video_processor import VideoProcessor


def get_openpose_params(model_folder):
    params = dict()
    # params["logging_level"] = 1
    # params["net_resolution"] = "-1x368" # default
    params["model_pose"] = "BODY_25"
    params["alpha_pose"] = 0.6
    # params["scale_gap"] = 0.3
    # params["scale_number"] = 1
    # params["render_threshold"] = 0.05
    # params["num_gpu_start"] = 0  # If GPU version is built, and multiple GPUs are available, set the ID here
    # params["disable_blending"] = False
    params["model_folder"] = model_folder
    return params


def main(argv):
    LoggingConfig.setup()

    input_files_path = "/Users/allarviinamae/EduWorkspace/master-thesis-training-videos/backflips"
    op_models_path = "/Users/allarviinamae/EduWorkspace/openpose/models"
    show_video = False

    try:
        opts, args = getopt.getopt(argv, "hi:m:s", ["inputFilesPath=", "opModelsPath=", "showVideo="])
    except getopt.GetoptError:
        logging.info('main.py -i <inputFilesPath> -m <opModelsPath> -s')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            logging.info('main.py -i <inputFilesPath> -m <opModelsPath> -s')
            sys.exit()
        elif opt in ("-i", "--inputFilesPath"):
            input_files_path = arg
        elif opt in ("-m", "--opModelsPath"):
            op_models_path = arg
        elif opt in ("-s", "--showVideo"):
            show_video = True

    logging.info(f'Input files path is {input_files_path}')
    logging.info(f'OpenPose models path is {op_models_path}')
    logging.info(f'Show video is {show_video}')

    try:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        # sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the
        # OpenPose/python module from there. This will install OpenPose and the python library at your desired
        # installation path. Ensure that this is in your python path in order to use it. sys.path.append(
        # '/usr/local/python')
        from openpose import pyopenpose as op
    except ImportError as e:
        logging.warn(
            'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python '
            'script in the right folder?')
        raise e

    # Initializing Python OpenPose wrapper. Constructing OpenPose object allocates GPU memory
    logging.info("Starting OpenPose Python Wrapper...")
    op_wrapper = op.WrapperPython()
    openpose_params = get_openpose_params(op_models_path)
    op_wrapper.configure(openpose_params)
    op_wrapper.start()
    logging.info("OpenPose Python Wrapper started")

    video_processor = VideoProcessor(op_wrapper, show_video)

    input_files = InputFileService.get_input_files(input_files_path)
    input_files.sort()

    for video_to_process in input_files:
        video_processor.process(video_to_process)


if __name__ == '__main__':
    main(sys.argv[1:])
