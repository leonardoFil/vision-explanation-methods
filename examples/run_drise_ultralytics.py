from pathlib import Path
import torch

from ultralytics import YOLO
from vision_explanation_methods import DRISE_runner, UltralyticsYoloWrapper

# --- config ---
IMAGE = r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Documents\GitHub\dolphin_detection\results_yolo11_inf\frame_000011.jpeg"
WEIGHTS = r"C:\Users\LFilipe\OneDrive - WavEC Offshore Renewables\Documents\GitHub\dolphin_detection\model\M5.pt"     # e.g., yolov8l.pt or your custom .pt
SAVE   = "drise_out.png"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 1  # set to your model's class count (or len(model.names))

# --- model + wrapper ---
yolo = YOLO(WEIGHTS)            # loads model and class names
wrapper = UltralyticsYoloWrapper(model=yolo, device=DEVICE, conf_threshold=0.0)

# --- run D-RISE ---
# The runner loads the image path itself and handles masking/visualization.
# You can tune nummasks/maskres for speed vs quality.
figs, out_path, labels = DRISE_runner.get_drise_saliency_map(
    imagelocation=IMAGE,
    model=wrapper,
    numclasses=NUM_CLASSES,
    savename=SAVE,
    nummasks=200,              # 25 is default; 200–1000 yields smoother maps
    maskres=(16, 16),          # coarser->faster; finer->slower
    maskpadding=None,
    devicechoice=DEVICE,
    max_figures=None
)

print(f"Saved: {out_path} | Labels: {labels}")
