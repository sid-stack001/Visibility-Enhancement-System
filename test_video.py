import os
import torch
import numpy as np
from options.test_options import TestOptions
from models import create_model
from util import util
import cv2
from datetime import datetime

# Define hard-coded parameters and paths
video_path = r"sd_video_2.mp4"  # Path to the input video
results_dir = './results/'            # Directory to save results
model_name = 'desmoke'  # Model name
model_type = 'test'                   # Model type
no_dropout = True                     # No dropout flag

if __name__ == '__main__':
    # Create TestOptions object and set parameters
    opt = TestOptions().parse()
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    opt.results_dir = results_dir  # Set results directory
    opt.name = model_name   # Set model name
    opt.model = model_type  # Set model type
    if no_dropout:
        opt.no_dropout = True  # Set no dropout flag
    else:
        opt.no_dropout = False

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_video_path = os.path.join(results_dir, f'output_video_{timestamp}.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Using XVID codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width * 2, frame_height))
    
    data = {"A": None, "A_paths": None}
    while True:
        success, input_image = webcam.read()
        if not success:
            print("Could not get an image. Please check your video source")
            break

        cv2.imshow("cam-input", input_image)

        input_image = cv2.resize(input_image, (256, 256))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = np.asarray([input_image])
        input_image = np.transpose(input_image, (0, 3, 1, 2))

        data['A'] = torch.FloatTensor(input_image)

        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        result_image = model.get_current_visuals()['fake']
        #print(result_image)
        result_image = util.tensor2im(result_image)
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = cv2.resize(result_image, (512, 512))

        cv2.imshow("cam-output", result_image)

        k = cv2.waitKey(1)
        if k == 27 or k == ord('q'):
            break

    cv2.destroyWindow("cam-input")
    cv2.destroyWindow("cam-output")
