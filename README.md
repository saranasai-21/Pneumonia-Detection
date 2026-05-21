# Pneumonia Detection using Deep Learning

A deep learning-based web application for detecting **Pneumonia from Chest X-ray images** using Convolutional Neural Networks (CNNs) and Transfer Learning techniques. This project leverages TensorFlow/Keras and FastAPI to provide an easy-to-use interface for medical image classification.

---

## Features

- Upload Chest X-ray images for prediction
- Detects whether the patient has:
  - **Pneumonia**
  - **Normal**
- Deep Learning model using **DenseNet / CNN**
- FastAPI-powered backend
- Interactive frontend using HTML templates
- Real-time predictions
- Easy deployment using Docker or local environment

---

## Tech Stack

- **Python**
- **TensorFlow / Keras**
- **FastAPI**
- **Jinja2 Templates**
- **HTML/CSS**
- **Pillow (PIL)**
- **NumPy**

---

## Project Structure

```bash
Pneumonia-Detection/
│
├── app.py                 
├── models/
│   └── pneumonia_model.keras  # Trained deep learning model
│
├── templates/
│   └── index.html         
│
├── static/                 
│
├── requirements.txt
├── Dockerfile
├── Metrics visualization.py
├── train.py
├── gradcam.py
└── README.md
```

---

## Dataset

The model is trained on the **Chest X-ray Images (Pneumonia)** dataset available on Kaggle.
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Dataset contains:
- Normal chest X-rays
- Pneumonia chest X-rays

Dataset structure:

```bash
train/
    NORMAL/
    PNEUMONIA/

test/
    NORMAL/
    PNEUMONIA/

val/
    NORMAL/
    PNEUMONIA/
```
# Parameters
<img width="527" height="242" alt="Screenshot 2026-05-22 035826" src="https://github.com/user-attachments/assets/0327ac2a-6805-414e-87dc-ecd968ce54c0" />

---

## Model Architecture

This project uses a **Transfer Learning-based CNN model** for feature extraction and classification.

Possible architectures used:
- DenseNet121
- ResNet50
- Custom CNN

The model is trained using:
- Data Augmentation
- Transfer Learning
- Binary Classification

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/saranasai-21/Pneumonia-Detection.git
cd Pneumonia-Detection
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

#### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
uvicorn app:app --reload
```

Application runs at:

```bash
http://127.0.0.1:8000
```
## Docker Setup

### Build Docker Image

```bash
docker build -t pneumonia-detection .
```

### Run Docker Container

```bash
docker run -p 8000:8000 pneumonia-detection
```

---

# Classification Report, ROC curve & Confusion Matrix
<img width="605" height="384" alt="Screenshot 2026-05-22 035720" src="https://github.com/user-attachments/assets/7628bfed-8a9d-4e41-bef5-808abb4824b8" />
<img width="529" height="470" alt="Screenshot 2026-05-22 040252" src="https://github.com/user-attachments/assets/7ce8725c-1e60-4622-b25c-6403d2664ad4" />
<img width="654" height="551" alt="Screenshot 2026-05-22 040419" src="https://github.com/user-attachments/assets/b466ba0a-2a38-4ac4-b300-a0042deabb6d" />

# Explainable AI (Grad-CAM)

Implemented Grad-CAM visualization for DenseNet121 to highlight important lung regions influencing pneumonia predictions with help of AI.

<img width="1083" height="697" alt="Screenshot 2026-05-22 041412" src="https://github.com/user-attachments/assets/4deb7df6-9d7d-49c5-9cf0-1dc9f345c4d7" />

### Live Demo Link : https://pneumonia-detection-druh.onrender.com/
## Screenshots
### Home Page
<img width="1187" height="584" alt="image" src="https://github.com/user-attachments/assets/f8637957-964f-4257-a691-5a204b441be5" />


### Prediction Result
<img width="900" height="849" alt="image" src="https://github.com/user-attachments/assets/533e06ee-6401-4464-8dd8-1f85ea2538e0" />


---

GitHub Repository:  
https://github.com/saranasai-21/Pneumonia-Detection

