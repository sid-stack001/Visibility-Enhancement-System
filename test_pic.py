import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import cv2
from datetime import datetime

# Define hard-coded parameters and paths
image_path = r"input_images\buddha.jpg"  # Path to the input image
results_dir = './results/'  # Directory to save results
model_name = 'desmoke'  # Model name
model_type = 'test'  # Model type
no_dropout = True  # No dropout flag

if __name__ == '__main__':
    # Create TestOptions object and set parameters
    opt = TestOptions().parse()
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.results_dir = results_dir  # Set results directory
    opt.name = model_name  # Set model name
    opt.model = model_type  # Set model type
    if no_dropout:
        opt.no_dropout = True  # Set no dropout flag
    else:
        opt.no_dropout = False

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    # Load and process a single image
    input_image = cv2.imread(image_path)

    if input_image is None:
        print("Could not read the image. Please check your image path.")
        exit()

    # Display the input image
    cv2.imshow("input-image", input_image)

    # Preprocess the image without resizing
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image_rgb = np.asarray([input_image_rgb])
    input_image_rgb = np.transpose(input_image_rgb, (0, 3, 1, 2))
    data = {"A": torch.FloatTensor(input_image_rgb), "A_paths": [image_path]}

    model.set_input(data)  # unpack data from data loader
    model.test()  # run inference

    result_image = model.get_current_visuals()['fake']
    result_image = util.tensor2im(result_image)
    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

    # Generate unique filenames using timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(results_dir, f'result_image_{timestamp}.png')

    # Save the result image
    cv2.imwrite(result_path, result_image)
    print(f"Result saved to {result_path}")

    # Ensure both images have the same height before fusing
    input_image_height, input_image_width = input_image.shape[:2]
    result_image_height, result_image_width = result_image.shape[:2]

    if input_image_height != result_image_height:
        # Resize result_image to match input_image height
        result_image = cv2.resize(result_image, (input_image_width, input_image_height))

    # Fuse the input image with the result image side-by-side
    fused_image = np.hstack((input_image, result_image))
    fused_image_path = os.path.join(results_dir, f'input_output_fused_{timestamp}.png')

    # Save the fused image
    cv2.imwrite(fused_image_path, fused_image)
    print(f"Fused image saved to {fused_image_path}")

    # Display the result and fused images
    cv2.imshow("output-image", result_image)
    cv2.imshow("fused-image", fused_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
