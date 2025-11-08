# 🧠 CT Trainer (Tkinter + MONAI)

A **desktop application** for training and predicting 3D CT volumes using **PyTorch** and **MONAI**, with a **user-friendly Tkinter GUI**. Handles DICOM datasets, offers real-time training progress, slice visualization, and dataset repair.

---

## Features

- ✅ Load and repair corrupted or missing DICOM files.
- ✅ Train a 3D CNN (ResNet-18) on CT volumes with **train/test split**.
- ✅ Save and load trained models.
- ✅ Predict CT scans from a folder and display probabilities.
- ✅ Interactive slice viewer for volume visualization.
- ✅ Thread-safe logging in GUI.
- ✅ Progress bars for training and prediction.
- ✅ Fully configurable through `config.json`.

---

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dataset Structure](#dataset-structure)
- [GUI Overview](#gui-overview)
- [Training](#training)
- [Prediction](#prediction)
- [Repair Dataset](#repair-dataset)
- [License](#license)

---

## Requirements

- Python 3.10+
- PyTorch
- MONAI
- pydicom
- numpy
- scikit-learn
- Pillow
- Tkinter (usually comes with Python)

Install required packages:

```bash
pip install torch torchvision torchaudio
pip install monai pydicom numpy scikit-learn pillow
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/ct-trainer.git
cd ct-trainer
Run the application:

bash
Copy code
python ct_trainer.py
Configuration
All settings are stored in config.json. Defaults:

json
Copy code
{
    "train_dir": "",
    "test_dir": "",
    "epochs": 4,
    "batch_size": 2,
    "learning_rate": 0.0001,
    "model_path": "model.pth",
    "classes_path": "classes.txt"
}
train_dir: Path to training dataset.

test_dir: Path to testing dataset.

epochs: Number of training epochs.

batch_size: Training batch size.

learning_rate: Learning rate for optimizer.

model_path: Path to save/load model.

classes_path: File storing class labels.

Dataset Structure
The dataset should follow this structure:

Copy code
dataset_root/
├── ClassA/
│   ├── Patient1/
│   │   ├── slice1.dcm
│   │   ├── slice2.dcm
│   │   └── ...
│   └── Patient2/
├── ClassB/
│   └── Patient1/
│       └── ...
Each class folder contains patient subfolders.

Each patient folder contains DICOM slices (.dcm).

GUI Overview
Train Dir / Test Dir: Browse to select dataset folders.

🧠 Train Model: Train 3D CNN on selected datasets.

📂 Load Saved Model: Load a previously trained model.

🛠️ Repair Dataset: Attempt to repair broken or missing DICOM metadata.

🔍 Predict on Folder: Run prediction on a folder of DICOM slices.

Slice Viewer: Scroll through 3D volume slices using the slider.

Log Window: Real-time training and prediction logs.

Progress Bars: Show training and prediction progress.

Exit: Close the application.

Training
Select your Train Dir and Test Dir.

Click 🧠 Train Model.

Training progress will display in the progress bar and log window.

Model and class labels are saved to model.pth and classes.txt.

Prediction
Click 🔍 Predict on Folder.

Select a folder containing DICOM slices.

Prediction probability (%) will display in the progress bar.

Scroll through slices using the slider for visualization.

Result label shows the predicted class.

Repair Dataset
Click 🛠️ Repair Dataset to:

Detect missing DICOM metadata.

Repair corrupted files if possible.

Log readable files count.

License
This project is licensed under the MIT License. See LICENSE/MIT.txt for details.

Author: Tarek Ahmadieh
Version: 1.0