import cv2
import socket
import numpy as np
import pickle
import torch
from options.test_options import TestOptions
from models import create_model
from util import util
from ultralytics import YOLO
import time

# === CONFIG ===
model_name = 'derain'
model_type = 'test'
no_dropout = True
results_dir = './results/'
TARGET_CLASS = 0  # person class

# === YOLO SETUP ===
yolo_model = YOLO("yolov8n.pt")

# === Desmoke Model Setup ===
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

# === Server Setup ===
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP protocol
ip = "192.168.127.114"  # Server IP
port = 2323  # Server Port
s.bind((ip, port))  # Bind IP and port to server

prev_time = 0

# === Make windows resizable ===
cv2.namedWindow("VR View", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("VR View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Canny Edge View", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Canny Edge View", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.namedWindow("Original vs Processed", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Original vs Processed", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    x = s.recvfrom(100000000)  # Receive data from client
    data = x[0]  # Extract data from the received message
    data = pickle.loads(data)  # Deserialize data

    # Decode the received frame
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is None:
        print("Received frame is empty, skipping...")
        continue

    original_frame = frame.copy()

    # === Desmoke (Image Processing) ===
    input_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
    input_tensor = np.asarray([input_rgb])
    input_tensor = np.transpose(input_tensor, (0, 3, 1, 2)).astype(np.float32) / 255.0
    data = {"A": torch.FloatTensor(input_tensor), "A_paths": ["frame.jpg"]}

    with torch.no_grad():
        model.set_input(data)
        model.test()
        result_image = model.get_current_visuals()['fake']
        processed = util.tensor2im(result_image)
        processed = cv2.cvtColor(np.array(processed), cv2.COLOR_RGB2BGR)

    # Resize if needed
    if processed.shape[:2] != original_frame.shape[:2]:
        processed = cv2.resize(processed, (original_frame.shape[1], original_frame.shape[0]))

    # === Canny Edge Detection ===
    gray_orig = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    blur_orig = cv2.GaussianBlur(gray_orig, (5, 5), 1.5)
    canny_orig = cv2.Canny(blur_orig, 100, 200)
    canny_orig = cv2.cvtColor(canny_orig, cv2.COLOR_GRAY2BGR)

    gray_processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    blur_processed = cv2.GaussianBlur(gray_processed, (5, 5), 1.5)
    canny_processed = cv2.Canny(blur_processed, 100, 200)
    canny_processed_colored = cv2.cvtColor(canny_processed, cv2.COLOR_GRAY2BGR)

    # === YOLOv8 Person Detection on Processed Image ===
    results = yolo_model(processed, verbose=False)[0]
    for box in results.boxes:
        if int(box.cls[0]) == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(processed, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(processed, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === YOLO on Canny Processed Image ===
    for box in results.boxes:
        if int(box.cls[0]) == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(canny_processed_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canny_processed_colored, "Person", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # === FPS Calculation ===
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(processed, f"FPS: {fps:.1f}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # === VR View: Side-by-Side Stereo ===
    stereo_frame = np.hstack((processed, processed))

    # === Canny Edge View (Side-by-Side) ===
    canny_edge_frame = np.hstack((canny_processed_colored, canny_processed_colored))

    # === Original vs Processed with Canny (Comparison View) ===
    top_row = np.hstack((original_frame, processed))
    bottom_row = np.hstack((canny_orig, canny_processed_colored))
    comparison_frame = np.vstack((top_row, bottom_row))

    # Show all three frames
    cv2.imshow("VR View", stereo_frame)
    cv2.imshow("Canny Edge View", canny_edge_frame)
    cv2.imshow("Original vs Processed", comparison_frame)

    # Exit if 'Enter' key is pressed
    if cv2.waitKey(10) == 13:
        break

cv2.destroyAllWindows()
s.close()
