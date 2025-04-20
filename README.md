# UNMASKING-FACES-USING-ECGANS
🎭 Unmasking Faces Using ECGAN
A Deep Learning-based project for reconstructing and unmasking occluded or partially hidden facial images using an Enhanced Conditional Generative Adversarial Network (ECGAN).

🧠 Overview
This project implements an ECGAN model designed to restore occluded or masked faces with high accuracy, preserving facial identity and structure. The model leverages adversarial training with a conditional setup to generate realistic unmasked versions of the input faces.

🚀 Features
✅ Mask removal and facial inpainting using ECGAN

🧬 Robust training with conditional GAN architecture

📊 Evaluation on standard datasets with PSNR, SSIM, and L1 Loss

📁 Clean and modular code structure for training, testing, and validation

📸 Visual outputs to compare masked vs. unmasked faces

🛠️ Tech Stack
Python 🐍

TensorFlow / PyTorch

OpenCV

NumPy, Matplotlib

Jupyter Notebook

📦 Dataset
We used a combination of publicly available datasets such as:

CelebA Dataset

LFW (Labeled Faces in the Wild)
Note: Preprocessing was applied to align and mask faces.
🔧 Installation
```bash

Clone the repository
git clone https://github.com/yourusername/ecgan-unmasking-faces.git
cd ecgan-unmasking-faces

Install required packages
pip install -r requirements.txt
```

🧪 Training & Evaluation
```bash

To train the ECGAN model
python train.py

To evaluate the trained model
python evaluate.py
```
