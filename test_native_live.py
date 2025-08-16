import os
import torch
import numpy as np
import cv2
import time
from options.test_options import TestOptions
from models import create_model
from util import util
from ultralytics import YOLO

# === CONFIG ===
model_name = 'desmoke'
model_type = 'test'
no_dropout = True
results_dir = './results/'
TARGET_CLASS = 0  # person class

# === YOLO SETUP ===
yolo_model = YOLO("yolov8n.pt")

if __name__ == '__main__':
    # Init desmoke model
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.results_dir = results_dir
    opt.name = model_name
    opt.model = model_type
    opt.no_dropout = no_dropout
    opt.eval = True

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    if os.path.isfile(opt.videosource):
        src = os.path.abspath(opt.videosource)
    else:
        src = int(opt.videosource)

    webcam = cv2.VideoCapture(src)
    cv2.namedWindow("cam-input")
    cv2.namedWindow("cam-output")
    
    data = {"A": None, "A_paths": None}
    while True:
        success, input_image = webcam.read()
        if not success:
            print("Could not get an image. Please check your video source")
            break

        cv2.imshow("cam-input", input_image)

        # input_image = cv2.resize(input_image, (256, 256))
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