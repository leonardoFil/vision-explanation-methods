from dataclasses import dataclass
from typing import List, Optional

import torch
from torch import Tensor

from ultralytics import YOLO

# Reuse the core types/utilities used by D-RISE
from ..explanations.common import DetectionRecord, GeneralObjectDetectionModelWrapper


@dataclass
class UltralyticsYoloWrapper(GeneralObjectDetectionModelWrapper):
    """
    Adapter to use an Ultralytics YOLO model (v8/v11) with D-RISE.

    Expected input to `predict` is a CHW float tensor in [0,1] on the target device.
    Returns a list with one DetectionRecord per image (we currently support single image).
    """
    model: YOLO
    conf_threshold: float = 0.0      # let D-RISE see low-conf too; you can raise it
    iou_threshold: float = 0.7       # NMS threshold used by Ultralytics call
    max_det: int = 300
    device: Optional[str] = None     # e.g. "cuda:0" or "cpu"

    def __post_init__(self):
        if self.device is not None:
            self.model.to(self.device)
        self.model.fuse() if hasattr(self.model, "fuse") else None

    @torch.no_grad()
    def predict(self, image_tensor: Tensor) -> List[DetectionRecord]:
        """
        Run the wrapped YOLO on a single image tensor (C,H,W) in [0,1].
        Returns [DetectionRecord] for D-RISE.
        """
        # --- accept (C,H,W) or (1,C,H,W) float in [0,1]
        if image_tensor.dim() == 4:
            image_tensor = image_tensor[0]
        assert image_tensor.dim() == 3, "Expected CHW tensor"

        # Keep original H,W for DetectionRecord
        _, H, W = image_tensor.shape

        # Convert to NumPy HWC uint8 to let Ultralytics handle resize/letterbox
        img_np = (image_tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")

        # Call Ultralytics with NumPy image; let it choose imgsz/stride internally
        results = self.model.predict(
            source=img_np,                   # HWC uint8
            verbose=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
            device=self.device if self.device is not None else None
        )

        # Build DetectionRecord (xyxy boxes, scores, labels)
        dets = results[0].boxes  # first (and only) image
        if dets is None or dets.shape[0] == 0 or dets.xyxy is None:
            # empty prediction
            h, w = H, W
            return [DetectionRecord(
                boxes=torch.zeros((0, 4), dtype=torch.float32),
                scores=torch.zeros((0,), dtype=torch.float32),
                labels=torch.zeros((0,), dtype=torch.long),
                image_size=(h, w)
            )]

        # Tensors on CPU for downstream ops in v-e-m
        xyxy  = dets.xyxy.detach().cpu().to(torch.float32)
        conf  = dets.conf.detach().cpu().to(torch.float32)
        clsid = dets.cls.detach().cpu().to(torch.long)

        D = xyxy.shape[0]
        C = getattr(self.model.model, 'nc', int(clsid.max().item()) + 1)

        # One-hot class scores
        class_scores = torch.zeros((D, C), dtype=torch.float32)
        if D > 0:
            class_scores[torch.arange(D), clsid] = conf

        record = DetectionRecord(
            bounding_boxes=xyxy,
            objectness_scores=conf,
            class_scores=class_scores,
        )
        return [record]
