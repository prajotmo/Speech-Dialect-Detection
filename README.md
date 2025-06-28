# Speech-Dialect-Detection

# 🗣️ Speech Dialect Detection in Marathi using RNN and CNN

This project focuses on detecting dialects in Marathi speech using deep learning techniques, specifically Recurrent Neural Networks (RNNs) and Convolutional Neural Networks (CNNs). It aims to classify speech samples into regional dialect categories to support language research and enhance voice-based interfaces in local Indian languages.

---

## 📌 Table of Contents

- [About](#about)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 📖 About

Marathi is spoken across different regions in Maharashtra, each with slight variations in dialect and pronunciation. This project uses deep learning models to classify audio clips based on their dialectal features.

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- Librosa (for audio processing)
- NumPy / Pandas
- Matplotlib / Seaborn (for visualization)

---

## 🎧 Dataset

- **Language**: Marathi
- **Input**: Raw audio (.wav) files
- **Classes**: Various dialects of Marathi (e.g., Pune, Vidarbha, Marathwada, etc.)
- Dataset may be collected manually or sourced from public Marathi speech corpora.

> Note: You may need to preprocess audio (resampling, trimming, MFCC extraction, etc.).

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/marathi-dialect-detection.git
cd marathi-dialect-detection

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
# Run training
python train_model.py

# Run inference on a sample audio
python predict.py --file sample.wav
