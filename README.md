### 🌫️ Visibility Enhancement System

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

## 📂 Dataset  

| Dataset   | Train Images | Test Images | Total |
|-----------|--------------|-------------|-------|
| REVIDE    | 3310         | 194         | 3504  |
| Rain1400  | 3600         | 400         | 4000  |
| Dense Haze| 80           | 30          | 110   |
| **Total** | **6990**     | **624**     | **7614** |

---

📊 Results
<img width="790" height="356" alt="image" src="https://github.com/user-attachments/assets/b12abb72-9834-4ea8-9402-be6797df3059" />

## 📈 Evaluation Metrics

PSNR (Peak Signal-to-Noise Ratio) - 28.09
SSIM (Structural Similarity Index) - 0.78

tested on SOTS indoor dataset

🛠️ Tech Stack

🐍 Python 3.9+

🔥 PyTorch

🎥 OpenCV

⚡ CUDA/cuDNN

## 📢 Future Work

🌌 Extending to fog, snow, and low-light scenarios

📱 Developing a lightweight version for mobile & edge devices

🤖 ROS integration for autonomous systems

🧩 Exploring transformers and diffusion-based approaches

Codebase to be uploaded soon

