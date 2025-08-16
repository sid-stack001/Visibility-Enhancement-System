# Visibility Enhancement System

Real-Time Dehazing & Deraining using CycleGAN

![WhatsApp Image 2025-08-16 at 14 50 31_e451d44f](https://github.com/user-attachments/assets/3111af34-0818-4a73-8a8d-6bc296bd94f7)

## 📌 Overview

Weather conditions like **haze** and **rain** degrade visibility, reducing the reliability of computer vision systems used in:

- 🚗 **Autonomous navigation**
- 📹 **Surveillance**
- 🛰️ **Remote sensing**
- 🎥 **Image & video enhancement**

This project introduces a **CycleGAN-based system** for **high-resolution multi-weather degradation removal**. Unlike conventional dehazing/deraining approaches, our model:

- Operates on **native image resolution** (avoiding downscaling)
- Handles **both haze & rain simultaneously**
- Achieves **competitive state-of-the-art results** on benchmark datasets

---

## ✨ Key Contributions

- 🔹 **Custom Dataset**: 7,614 unpaired images (synthetic + real) from REVIDE, Rain1400, and Dense-Haze
- 🔹 **Architecture Modifications**: CycleGAN tailored for **high-fidelity restoration**
- 🔹 **Training Optimizations**: ADAM optimizer, 100 epochs, unpaired image learning
- 🔹 **Benchmark Testing**: Evaluated on **SOTS** and **O-HAZE** datasets
- 🔹 **Results**: Achieved **PSNR ~29 dB** and competitive **SSIM scores**

---

## 🧠 Methodology

🔹 Why CycleGAN?

CycleGAN enables image-to-image translation without paired datasets, making it ideal for visibility enhancement where paired clean-weather images are hard to obtain.

🔹 Architecture

Generator: Translates hazy/rainy frames into clear frames

Discriminator: Ensures generated outputs are visually realistic

Cycle Consistency Loss: Maintains structural fidelity between degraded and restored images.
<img width="692" height="531" alt="image" src="https://github.com/user-attachments/assets/460a0ba3-60d6-4e07-a263-08a106a9c352" />

## 📂 Project Structure  

```bash
├── canny_edge_det/      # Checks image features through canny edges
├── checkpoints/         # Saved model checkpoints  
├── data/                # Datasets (train/test splits)  
├── datasets/            # Dataset processing scripts  
├── docs/                # Documentation and related resources  
├── imgs/                # Images for README/demo  
├── input_images/        # Input images for testing  
├── input_videos/        # Input videos for testing
├── models/              # Stores models
├── onnx/                # Converts model to .onnx format
├── options/             # Configuration files and options  
├── results/             # Generated results (images/videos)  
├── results_past/        # Previous experiment results  
├── scripts/             # Utility scripts for training/inference  
├── util/                # Helper functions and utilities  
│
├── LICENSE              # License file (MIT)  
├── README.md            # Project documentation  
├── desktop.ini          # Windows system file (can be ignored)  
├── server.py            # API/Server script  
├── test.py              # Testing script  
├── test_native_live.py  # Live webcam inference (native resolution)  
├── test_pic.py          # Single image test script  
├── test_resize.py       # Test with resized inputs  
├── test_server.py       # Server-based test script  
├── test_video.py        # Video file inference  
├── test_video_live.py   # Live video stream inference  
├── train.py             # Training entry point  
├── yolov8.pt            # YOLOv8 weights (for auxiliary tasks if used)  

```
## 📂 Dataset

| Dataset    | Train Images | Test Images | Total    |
| ---------- | ------------ | ----------- | -------- |
| REVIDE     | 3310         | 194         | 3504     |
| Rain1400   | 3600         | 400         | 4000     |
| Dense Haze | 80           | 30          | 110      |
| **Total**  | **6990**     | **624**     | **7614** |

---


## 🏁 Getting Started  

Follow these steps to quickly run the Visibility Enhancement System on your machine.  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/visibility-enhancement.git
cd visibility-enhancement
```

### 2️⃣ Setup Environment

It’s recommended to use Conda or virtualenv.
```
conda create -n vis-enhance python=3.9
conda activate vis-enhance
pip install -r requirements.txt
```

### 4️⃣ Run Inference

On a single image:
```
python test_pic.py --input input_images/sample.jpg --output results/output.jpg
```
On a video file:
```
python test_video.py --input input_videos/sample.mp4 --output results/output.mp4
```
Live webcam demo (native resolution):
```
python test_native_live.py

```
---

## 📊 Results

<img width="975" height="348" alt="image" src="https://github.com/user-attachments/assets/c0864a59-5498-410d-9b49-98455d7d2b66" />
<img width="975" height="296" alt="image" src="https://github.com/user-attachments/assets/7055dfe5-16ae-428e-a5dc-20b0d8089890" />
<img width="975" height="313" alt="image" src="https://github.com/user-attachments/assets/d87771c2-973d-4079-8370-612d4f8eff86" />

## 📈 Evaluation Metrics

- PSNR (Peak Signal-to-Noise Ratio) - 28.09
- SSIM (Structural Similarity Index) - 0.78

Tested on the SOTS indoor dataset.

## 🛠️ Tech Stack

🐍 Python 3.9+

🔥 PyTorch

🎥 OpenCV

⚡ CUDA/cuDNN

## 📢 Future Work

🌌 Extending to fog, snow, and low-light scenarios

📱 Developing a lightweight version for mobile & edge devices

🤖 ROS integration for autonomous systems

🧩 Exploring transformers and diffusion-based approaches

## 🙏 Acknowledgements  

This work is built upon the excellent open-source implementation of CycleGAN and Pix2Pix provided by [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).  
We sincerely thank the authors for their contributions to the research community. 
