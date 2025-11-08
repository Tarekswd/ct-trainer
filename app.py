# -------------------------------------------------------
# 🧠 CT Trainer (Tkinter + MONAI)
# ✅ DICOM handling, thread-safe logging,
#    safe model loading, PDAC & training progress
# -------------------------------------------------------

import os, glob, json, torch, threading, numpy as np, pydicom
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import resnet18
from monai.transforms import Compose, ScaleIntensityRange, Resize
from sklearn.metrics import classification_report
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from queue import Queue
import ctypes

# Patch DWM (Desktop Window Manager) to avoid COMException
try:
    ctypes.windll.dwmapi.DwmIsCompositionEnabled
except Exception:
    ctypes.windll.dwmapi.DwmIsCompositionEnabled = lambda ptr: 0
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# VOI LUT fallback
try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    def apply_voi_lut(arr, ds):
        return arr

from pydicom.uid import ImplicitVRLittleEndian

# ------------------ Config ------------------
CONFIG_PATH = "config.json"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH) as f:
            config = json.load(f)
    except Exception:
        config = {}
else:
    config = {}

default_path = r"C:\Users\Admin\Desktop\python\manifest-1617826555824\Pseudo-PHI-DICOM-Data"

config.setdefault("train_dir", default_path)
config.setdefault("test_dir", default_path)
config.setdefault("epochs", 4)  # 4 epochs
config.setdefault("batch_size", 2)
config.setdefault("learning_rate", 1e-4)
config.setdefault("model_path", "model.pth")
config.setdefault("classes_path", "classes.txt")

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=4)

MODEL_PATH = config["model_path"]
CLASSES_PATH = config["classes_path"]

# ------------------ Logger ------------------
log_queue = Queue()
def log_callback(msg):
    log_queue.put(msg)

# ------------------ DICOM reader ------------------
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

# ------------------ Volume loader ------------------
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

# ------------------ Dataset ------------------
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

# ------------------ Model + transforms ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = Compose([
    ScaleIntensityRange(a_min=0, a_max=1, b_min=0, b_max=1, clip=True),
    Resize((64, 128, 128))
])

# ------------------ Model loader ------------------
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

# ------------------ Training ------------------
def train_model(train_dir, test_dir, log_cb):
    global model
    try:
        train_ds = DicomVolumeDataset(train_dir, transform)
        test_ds = DicomVolumeDataset(test_dir, transform)
        if len(train_ds) == 0:
            log_cb("❌ No training data found.")
            return
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=config["batch_size"])
        classes = train_ds.classes
        with open(CLASSES_PATH, "w") as f:
            f.write("\n".join(classes))
        model = resnet18(spatial_dims=3, n_input_channels=1, num_classes=len(classes)).to(device)
        crit = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters(), lr=config["learning_rate"])

        log_cb(f"🚀 Training for {config['epochs']} epochs...")
        for e in range(config["epochs"]):
            model.train()
            total = 0
            for v, l in train_loader:
                v, l = v.to(device), l.to(device)
                opt.zero_grad()
                o = model(v)
                loss = crit(o, l)
                loss.backward()
                opt.step()
                total += loss.item()
            log_cb(f"Epoch {e+1}/{config['epochs']} | Loss: {total/len(train_loader):.4f}")
            root.after(0, lambda e=e: train_bar.config(value=e+1))

        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for v, l in test_loader:
                v = v.to(device)
                o = model(v)
                p = torch.argmax(o, dim=1).cpu().numpy()
                y_pred.extend(p)
                y_true.extend(l.numpy())
        log_cb("✅ Training complete:\n" + classification_report(y_true, y_pred, target_names=classes))
        torch.save(model.state_dict(), MODEL_PATH)
        log_cb(f"💾 Model saved to {MODEL_PATH}")
        root.after(0, lambda: train_bar.config(value=0))
    except Exception as e:
        log_cb(f"❌ Training error: {e}")

# ------------------ Prediction ------------------
def predict_dicom(folder):
    if model is None:
        return "❌ Model not loaded", None
    vol, slices = load_volume_from_folder(folder)
    if vol is None:
        return "❌ No valid slices", None
    t = torch.tensor(vol, dtype=torch.float32)
    if transform:
        t = transform(t)
    t = t.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        pred_idx = int(torch.argmax(out, dim=1).item())
        pred_class = model.classes[pred_idx]
        pdac_prob = None
        if "PDAC" in model.classes:
            pdac_idx = model.classes.index("PDAC")
            pdac_prob = probs[pdac_idx]*100
        if pdac_prob is not None:
            result_text = f"🧩 Prediction: {pred_class} ({pdac_prob:.1f}% PDAC confidence)"
        else:
            result_text = f"🧩 Prediction: {pred_class} ({probs[pred_idx]*100:.1f}% confidence)"
    return result_text, slices

# ------------------ Repair ------------------
def repair_dataset(dirs, log_cb):
    fixed, total = 0, 0
    for root in dirs:
        if not root or not os.path.isdir(root):
            continue
        for subdir, _, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(".dcm"):
                    continue
                total += 1
                if load_and_process_dicom(os.path.join(subdir, f), repair=True) is not None:
                    fixed += 1
    log_cb(f"🔧 Repair complete — {fixed}/{total} readable")

# ------------------ GUI ------------------
root = tk.Tk()
root.title("🧠 CT Trainer")
root.geometry("900x1000")
root.configure(bg="#111111")

title = tk.Label(root, text="🧠 CT Trainer", fg="white", bg="#111111", font=("Segoe UI", 18, "bold"))
title.pack(pady=10)

log_text = tk.Text(root, height=18, width=110, bg="#0b0b0b", fg="#00ff88", font=("Consolas", 10))
log_text.pack(pady=8)

def process_log_queue():
    while not log_queue.empty():
        msg = log_queue.get_nowait()
        log_text.insert(tk.END, msg + "\n")
        log_text.see(tk.END)
    root.after(100, process_log_queue)
process_log_queue()

train_dir_var = tk.StringVar(value=config["train_dir"])
test_dir_var = tk.StringVar(value=config["test_dir"])

frame = tk.Frame(root, bg="#111111")
frame.pack(pady=6)

def browse_train():
    p = filedialog.askdirectory(title="Select TRAIN Folder")
    if p: train_dir_var.set(p)

def browse_test():
    p = filedialog.askdirectory(title="Select TEST Folder")
    if p: test_dir_var.set(p)

tk.Label(frame, text="Train Dir:", bg="#111111", fg="white").grid(row=0, column=0, sticky="w")
tk.Entry(frame, textvariable=train_dir_var, width=70).grid(row=0, column=1, padx=6)
tk.Button(frame, text="Browse", command=browse_train, bg="#333", fg="white").grid(row=0, column=2, padx=6)

tk.Label(frame, text="Test Dir:", bg="#111111", fg="white").grid(row=1, column=0, sticky="w")
tk.Entry(frame, textvariable=test_dir_var, width=70).grid(row=1, column=1, padx=6)
tk.Button(frame, text="Browse", command=browse_test, bg="#333", fg="white").grid(row=1, column=2, padx=6)

action_frame = tk.Frame(root, bg="#111111")
action_frame.pack(pady=8)

# Training progress bar
train_bar = ttk.Progressbar(root, length=480, maximum=config["epochs"])
train_bar.pack(pady=6)

# PDAC prediction bar + percentage beside it
pdac_frame = tk.Frame(root, bg="#111111")
pdac_frame.pack(pady=4)
pdac_bar = ttk.Progressbar(pdac_frame, length=280, maximum=100)
pdac_bar.pack(side="left", padx=(0, 6))
pdac_label = tk.Label(pdac_frame, text="0%", fg="white", bg="#111111", font=("Segoe UI", 10, "bold"))
pdac_label.pack(side="left")

# Slice viewer
canvas = tk.Label(root, bg="#111111")
canvas.pack(pady=6)

slice_slider = tk.Scale(root, from_=0, to=0, orient="horizontal", length=480, label="Slice", bg="#111111", fg="white")
slice_slider.pack(pady=6)

result_label = tk.Label(root, text="", fg="white", bg="#111111", font=("Segoe UI", 14))
result_label.pack(pady=4)

current_slices = None
def update_slice(val):
    global current_slices
    if current_slices is None: return
    idx = int(val)
    if idx < 0 or idx >= len(current_slices): return
    img = current_slices[idx]
    img = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img).resize((320, 320))
    tkimg = ImageTk.PhotoImage(pil)
    canvas.config(image=tkimg)
    canvas.image = tkimg

slice_slider.config(command=update_slice)

# ------------------ Buttons ------------------
def start_training():
    if not train_dir_var.get() or not test_dir_var.get():
        messagebox.showerror("Error", "Please select train and test folders!")
        return
    for btn in [train_btn, load_btn, predict_btn, repair_btn]:
        btn.config(state="disabled")
    train_btn.config(text="⏳ Training...")
    def run_training():
        try:
            train_model(train_dir_var.get(), test_dir_var.get(), log_callback)
        finally:
            def enable_buttons():
                train_btn.config(state="normal", text="🧠 Train Model")
                load_btn.config(state="normal")
                predict_btn.config(state="normal")
                repair_btn.config(state="normal")
                messagebox.showinfo("Training", "✅ Training complete!")
            root.after(0, enable_buttons)
    threading.Thread(target=run_training, daemon=True).start()

train_btn = tk.Button(action_frame, text="🧠 Train Model", command=start_training, bg="#2ecc71", fg="white", font=("Segoe UI", 12, "bold"))
train_btn.grid(row=0, column=0, padx=6)

load_btn = tk.Button(action_frame, text="📂 Load Saved Model", command=lambda: threading.Thread(target=load_saved_model, args=(log_callback,), daemon=True).start(), bg="#9b59b6", fg="white", font=("Segoe UI", 12, "bold"))
load_btn.grid(row=0, column=1, padx=6)

repair_btn = tk.Button(action_frame, text="🛠️ Repair Dataset", command=lambda: threading.Thread(target=repair_dataset, args=([train_dir_var.get(), test_dir_var.get()], log_callback), daemon=True).start(), bg="#f39c12", fg="white", font=("Segoe UI", 12, "bold"))
repair_btn.grid(row=0, column=2, padx=6)

def run_predict_worker(folder):
    log_callback("🔍 Preparing prediction...")
    result, slices = predict_dicom(folder)
    def gui_update():
        global current_slices
        result_label.config(text=result)

        # Update PDAC bar and label beside it
        pdac_bar['value'] = 0
        pdac_label.config(text="0%")
        if "PDAC" in getattr(model, "classes", []):
            pdac_idx = model.classes.index("PDAC")
            vol, _ = load_volume_from_folder(folder)
            if vol is not None:
                t = torch.tensor(vol, dtype=torch.float32)
                if transform: t = transform(t)
                t = t.unsqueeze(0).to(device)
                with torch.no_grad():
                    out = model(t)
                    probs = F.softmax(out, dim=1).cpu().numpy()[0]
                    pdac_percent = probs[pdac_idx]*100
                    pdac_bar['value'] = pdac_percent
                    pdac_label.config(text=f"{pdac_percent:.1f}%")  # beside the bar

        if slices is not None:
            current_slices = slices
            slice_slider.config(to=len(slices)-1)
            slice_slider.set(0)
            update_slice(0)

    root.after(0, gui_update)

def on_predict_button():
    folder = filedialog.askdirectory(title="Select DICOM Folder for Prediction")
    if not folder:
        return
    threading.Thread(target=run_predict_worker, args=(folder,), daemon=True).start()

predict_btn = tk.Button(action_frame, text="🔍 Predict on Folder", command=on_predict_button, bg="#3498db", fg="white", font=("Segoe UI", 12, "bold"))
predict_btn.grid(row=0, column=3, padx=6)

exit_btn = tk.Button(action_frame, text="Exit", command=root.quit, bg="#e74c3c", fg="white", font=("Segoe UI", 12, "bold"))
exit_btn.grid(row=0, column=4, padx=6)

# ------------------ Load model on startup ------------------
threading.Thread(target=lambda: load_saved_model(log_callback), daemon=True).start()

root.mainloop()
