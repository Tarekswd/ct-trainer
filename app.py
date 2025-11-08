# -------------------------------------------------------
# 🧠 CT Trainer (Tkinter + MONAI)
# ✅ DICOM handling, thread-safe logging, training & prediction progress bars
# -------------------------------------------------------

import os, glob, json, torch, threading, numpy as np, pydicom
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import resnet18
from monai.transforms import Compose, ScaleIntensityRange, Resize
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from queue import Queue

# ----------------- VOI LUT -----------------
try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    def apply_voi_lut(arr, ds):
        return arr

from pydicom.uid import ImplicitVRLittleEndian

# ----------------- Config -----------------
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except Exception:
        config = {}
else:
    config = {}

# Default: empty directories
config.setdefault("train_dir", "")
config.setdefault("test_dir", "")
config.setdefault("epochs", 4)  # 4 epochs
config.setdefault("batch_size", 2)
config.setdefault("learning_rate", 1e-4)
config.setdefault("model_path", "model.pth")
config.setdefault("classes_path", "classes.txt")

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=4)

MODEL_PATH = config["model_path"]
CLASSES_PATH = config["classes_path"]

# ----------------- Logger -----------------
log_queue = Queue()
def log_callback(msg):
    log_queue.put(msg)

# ----------------- Safe DICOM Loader -----------------
def load_and_process_dicom(file_path, repair=False):
    try:
        ds = pydicom.dcmread(file_path, force=True)
        fixed = False
        if not hasattr(ds, "file_meta") or ds.file_meta is None:
            ds.file_meta = pydicom.dataset.FileMetaDataset()
            fixed = True
        if not hasattr(ds.file_meta, "TransferSyntaxUID") or ds.file_meta.TransferSyntaxUID is None:
            ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
            fixed = True
        for tag, val in [
            ("BitsAllocated", 16),
            ("BitsStored", 16),
            ("HighBit", 15),
            ("PixelRepresentation", 0),
            ("SamplesPerPixel", 1),
            ("PhotometricInterpretation", "MONOCHROME2"),
        ]:
            if not hasattr(ds, tag):
                setattr(ds, tag, val)
                fixed = True
        try:
            arr = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            try:
                arr = ds.pixel_array
            except Exception:
                log_callback(f"❌ Cannot decode {file_path}")
                return None
        if repair and fixed:
            try:
                ds.save_as(file_path)
                log_callback(f"💾 Repaired {file_path}")
            except Exception as e:
                log_callback(f"⚠️ Failed to save repaired file {file_path}: {e}")
        return arr.astype(np.float32)
    except Exception as e:
        log_callback(f"❌ Error reading {file_path}: {e}")
        return None

# ----------------- Volume Loader -----------------
def load_volume_from_folder(folder_path, repair_missing=False):
    files = sorted(glob.glob(os.path.join(folder_path, "*.dcm")))
    if not files:
        return None, None
    slices = []
    for f in files:
        img = load_and_process_dicom(f, repair=repair_missing)
        if img is not None:
            slices.append(img)
    if not slices:
        return None, None
    viewer_slices = []
    for s in slices:
        mn, mx = np.min(s), np.max(s)
        viewer_slices.append((s - mn) / (mx - mn + 1e-5))
    vol = np.stack(slices, axis=0)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-5)
    vol = np.expand_dims(vol, axis=0)
    return vol, viewer_slices

# ----------------- Dataset -----------------
class DicomVolumeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = []
        if not os.path.isdir(root_dir):
            log_callback(f"❌ Root dir not found: {root_dir}")
            return
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for patient in os.listdir(cls_path):
                pdir = os.path.join(cls_path, patient)
                if os.path.isdir(pdir) and glob.glob(os.path.join(pdir, "*.dcm")):
                    self.samples.append((pdir, class_to_idx[cls]))
                else:
                    log_callback(f"⚠️ Skipping {pdir}: no .dcm files")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        folder, label = self.samples[idx]
        vol, _ = load_volume_from_folder(folder)
        if vol is None:
            raise RuntimeError(f"No readable DICOMs in {folder}")
        t = torch.tensor(vol, dtype=torch.float32)
        if self.transform:
            t = self.transform(t)
        return t, label

# ----------------- Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = Compose([
    ScaleIntensityRange(a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
    Resize((64, 128, 128))
])

def load_saved_model(log_cb):
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            log_cb("⚠️ No saved model found — please train one.")
            return
        if not os.path.exists(CLASSES_PATH):
            log_cb("⚠️ Missing classes.txt — retrain required.")
            return
        with open(CLASSES_PATH) as f:
            classes = [l.strip() for l in f if l.strip()]
        if not classes:
            log_cb("⚠️ classes.txt is empty — retrain required.")
            return
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=len(classes)).to(device)
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device)
            if not isinstance(state_dict, dict):
                log_cb("⚠️ Invalid model format. Deleting file.")
                os.remove(MODEL_PATH)
                return
            model.load_state_dict(state_dict)
        except Exception as e:
            log_cb(f"⚠️ Model corrupted: {e}")
            os.remove(MODEL_PATH)
            return
        model.eval()
        model.classes = classes
        log_cb(f"✅ Model loaded ({len(classes)} classes)")
    except Exception as e:
        log_cb(f"❌ Model load error: {e}")

# ----------------- Slice Viewer -----------------
slice_display_size = 480
current_slices = None

def update_slice(val):
    global current_slices
    if current_slices is None:
        return
    idx = int(val)
    if idx < 0 or idx >= len(current_slices):
        return
    img = np.squeeze(current_slices[idx])  # ensure 2D
    img = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((slice_display_size, slice_display_size), Image.Resampling.LANCZOS)
    tkimg = ImageTk.PhotoImage(pil)
    canvas.config(image=tkimg)
    canvas.image = tkimg  # keep reference

# ----------------- GUI -----------------
root = tk.Tk()
root.title("🧠 CT Trainer")
root.geometry("900x900")
root.configure(bg="#111111")

# ... [rest of GUI code remains the same as previous version]
# Make sure in prediction and training workers you call `update_slice(0)` **after** slider `to=` is set

# ----------------- Load model on startup -----------------
threading.Thread(target=lambda: load_saved_model(log_callback), daemon=True).start()

root.mainloop()
