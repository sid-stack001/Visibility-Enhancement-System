

import os
import torch
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util import util
import cv2
from datetime import datetime

# === Config ===
image_path = r"input_images/buddha.jpg"   # Path to input image
results_dir = './results/'                # Output dir
model_name = 'desmoke'                     # Model name
model_type = 'test'                        # Model type
no_dropout = True
resize_width = 512                         # Resize to this width
resize_height = 512                        # Resize to this height

# Force GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    # Create TestOptions
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

    # Create & load model
    model = create_model(opt)
    model.setup(opt)
    # model.to(device)   # send model to GPU
    if opt.eval:
        model.eval()

    # Load image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print("❌ Could not read the image. Check the path.")
        exit()

    # Resize BEFORE sending to model
    input_image = cv2.resize(input_image, (resize_width, resize_height))
    cv2.imshow("input-image", input_image)

    # Preprocess
    input_image_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image_rgb = np.asarray([input_image_rgb], dtype=np.float32)
    input_image_rgb = np.transpose(input_image_rgb, (0, 3, 1, 2))
    data = {"A": torch.from_numpy(input_image_rgb).to(device), "A_paths": [image_path]}

    # Check GPU memory usage before inference
    torch.cuda.synchronize()
    print(f"🔥 GPU Memory Before: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # Run inference
    model.set_input(data)
    model.test()

    # Check GPU memory usage after inference
    torch.cuda.synchronize()
    print(f"🔥 GPU Memory After: {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # Get result
    result_image = model.get_current_visuals()['fake']
    result_image = util.tensor2im(result_image)
    result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_path = os.path.join(results_dir, f'result_image_{timestamp}.png')
    cv2.imwrite(result_path, result_image)
    print(f"✅ Result saved to {result_path}")

    # Fuse side-by-side
    fused_image = np.hstack((input_image, cv2.resize(result_image, (resize_width, resize_height))))
    fused_image_path = os.path.join(results_dir, f'input_output_fused_{timestamp}.png')
    cv2.imwrite(fused_image_path, fused_image)
    print(f"✅ Fused image saved to {fused_image_path}")

    # Display
    cv2.imshow("output-image", result_image)
    cv2.imshow("fused-image", fused_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
