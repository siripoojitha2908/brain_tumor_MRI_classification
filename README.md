# 🧠 Brain Tumor MRI Classification Using Deep Learning

A deep learning project that classifies brain MRI images into four categories using a custom Convolutional Neural Network (CNN) and transfer learning with MobileNetV2. The project compares both models and provides an interactive web application built with Streamlit for image-based predictions.

## 📌 Project Overview

Brain tumor classification from MRI images is an important computer vision application in medical image analysis. This project explores how deep learning models can classify MRI images into different categories.

Two models are implemented and evaluated:

- **Custom CNN:** A convolutional neural network trained from scratch.
- **MobileNetV2:** A transfer learning model using ImageNet-pretrained weights.

MobileNetV2 achieved the better test accuracy and is used in the Streamlit application.

## 🎯 Objectives

- Classify brain MRI images into four categories.
- Build and train a custom CNN model.
- Apply transfer learning using MobileNetV2.
- Compare model performance using test accuracy and classification metrics.
- Visualize model performance using confusion matrices.
- Develop a web application for uploading MRI images and viewing predictions.

## 🧩 Classification Categories

The models classify images into the following four classes:

| Class | Description |
|---|---|
| Glioma | Glioma tumor category |
| Meningioma | Meningioma tumor category |
| No Tumor | Images labeled as having no tumor |
| Pituitary | Pituitary tumor category |

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Models:** Custom CNN, MobileNetV2
- **Image Processing:** Pillow, NumPy
- **Data Visualization:** Matplotlib
- **Evaluation:** Scikit-learn
- **Web Application:** Streamlit
- **Development Environment:** Visual Studio Code

## 📂 Project Structure

```text
brain_tumor_mri/
│
├── dataset/
│   ├── train/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   │
│   ├── valid/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── no_tumor/
│   │   └── pituitary/
│   │
│   └── test/
│       ├── glioma/
│       ├── meningioma/
│       ├── no_tumor/
│       └── pituitary/
│
├── models/
│   ├── custom_cnn.keras
│   ├── mobilenetv2.keras
│   └── class_names.json
│
├── outputs/
│   ├── cnn_accuracy.png
│   ├── transfer_accuracy.png
│   ├── cnn_confusion_matrix.png
│   └── mobilenetv2_confusion_matrix.png
│
├── src/
│   ├── train_cnn.py
│   ├── train_transfer.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The file tree represents the intended project organization. Keep only filenames that actually exist in your folder.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

Replace `YOUR_USERNAME` with your GitHub username and use the actual repository URL after creating it.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

If dependencies are not yet listed in `requirements.txt`, install the required packages:

```bash
python -m pip install tensorflow numpy matplotlib scikit-learn streamlit pillow
```

### 4. Prepare the dataset

Place the MRI images in the `dataset` directory, organized into `train`, `valid`, and `test` folders. Each split should contain the four class subfolders.

The training and validation scripts expect this folder structure.

### 5. Train the models

Train the custom CNN:

```bash
python src/train_cnn.py
```

Train the MobileNetV2 model:

```bash
python src/train_transfer.py
```

The scripts save trained models and training visualizations in the appropriate directories.

### 6. Evaluate the models

```bash
python src/evaluate.py
```

The evaluation script reports test accuracy, precision, recall, F1-score, and confusion matrices for both models.

### 7. Run the Streamlit application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

Upload a JPG, JPEG, or PNG MRI image and click **Analyze MRI Image** to view the model's predicted class and confidence scores.

## 📊 Model Performance

Both models were evaluated on the separate test dataset containing **246 MRI images** across four classes.

| Metric | Custom CNN | MobileNetV2 |
|---|---:|---:|
| Test Accuracy | 57.32% | **91.06%** |
| Macro Precision | 64% | 91% |
| Macro Recall | 55% | 91% |
| Macro F1-score | 52% | 91% |

### Key Findings

- MobileNetV2 achieved a test accuracy of **91.06%**.
- The custom CNN achieved a test accuracy of **57.32%**.
- MobileNetV2 outperformed the custom CNN by **33.74 percentage points** on this test set.
- Transfer learning provided stronger classification performance for this dataset.
- The MobileNetV2 model was selected for the Streamlit prediction interface.

### MobileNetV2 Classification Report

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Glioma | 93% | 96% | 94% |
| Meningioma | 88% | 81% | 84% |
| No Tumor | 91% | 88% | 90% |
| Pituitary | 91% | 98% | 95% |

The results are specific to the dataset used for this experiment. Performance may differ on images from other hospitals, scanners, or patient populations.

## 📈 Evaluation Visualizations

The `outputs/` directory contains training accuracy plots and confusion matrices generated by the project.

- `cnn_accuracy.png` — Custom CNN training history.
- `transfer_accuracy.png` — MobileNetV2 training history.
- `cnn_confusion_matrix.png` — Custom CNN test predictions.
- `mobilenetv2_confusion_matrix.png` — MobileNetV2 test predictions.

These images can be included below as project screenshots when presenting the repository.

## 🚀 Future Improvements

- Fine-tune selected MobileNetV2 layers.
- Apply data augmentation and class-balancing techniques where appropriate.
- Evaluate the model on independent external datasets.
- Add Grad-CAM visualizations to help interpret image regions influencing predictions.
- Improve the interface with clearer prediction summaries and downloadable reports.
- Explore model calibration and robustness across different imaging sources.

## ⚠️ Medical Disclaimer

**This project is intended for educational and research purposes only.**

The model is not a clinically validated diagnostic system. Its predictions and confidence scores must not be used to diagnose, rule out, or treat a medical condition. A high confidence score does not guarantee a correct prediction. Always consult qualified healthcare professionals for medical interpretation and diagnosis.

