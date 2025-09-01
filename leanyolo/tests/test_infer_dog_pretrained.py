import os
from pathlib import Path

import pytest
import cv2
import torch

from leanyolo.models import get_model, get_model_weights
from leanyolo.data.coco import coco80_class_names
from leanyolo.utils.letterbox import letterbox
from leanyolo.utils.box_ops import unletterbox_coords


def _weights_available_locally(model_name: str, filename: str) -> bool:
    # Check LEANYOLO_WEIGHTS_DIR
    env_dir = os.environ.get("LEANYOLO_WEIGHTS_DIR")
    if env_dir and os.path.exists(os.path.join(env_dir, filename)):
        return True
    # Check default cache
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "leanyolo")
    if os.path.exists(os.path.join(cache_dir, filename)):
        return True
    return False


@pytest.mark.skipif(
    not _weights_available_locally("yolov10s", "yolov10s.pt"),
    reason="Official weights not cached locally; set LEANYOLO_WEIGHTS_DIR or prepopulate ~/.cache/leanyolo",
)
def test_dog_bicycle_truck_present_with_high_confidence():
    img_path = Path("dog.jpg")
    assert img_path.exists(), "dog.jpg must exist in repo root"

    # Load image (BGR->RGB)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    assert bgr is not None, "Failed to read dog.jpg"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Prepare model + class names
    class_names = coco80_class_names()
    model = get_model(
        name="yolov10s",
        weights="PRETRAINED_COCO",
        class_names=class_names,
        input_norm_subtract=[0.0, 0.0, 0.0],
        input_norm_divide=[255.0, 255.0, 255.0],
    ).to("cpu").eval()

    # Inference pipeline similar to infer.py
    imgsz = 640
    lb_img, gain, pad = letterbox(rgb, new_shape=imgsz)
    x = torch.from_numpy(lb_img).permute(2, 0, 1).float().unsqueeze(0)

    with torch.inference_mode():
        model.post_conf_thresh = 0.01  # low; we filter later at 0.80
        model.post_iou_thresh = 0.65
        raw = model(x)
        dets = model.decode_forward(raw)[0][0]

    # Coordinates are not needed for class/score checks; avoid in-place ops on inference tensors

    # Verify presence of target classes with confidence >= 0.80
    want = {"dog": False, "bicycle": False, "truck": False}
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    target_idxs = {k: name_to_idx[k] for k in want.keys()}

    for i in range(dets.shape[0]):
        x1, y1, x2, y2, score, cls_idx = dets[i].tolist()
        if score >= 0.80:
            for cname, cidx in target_idxs.items():
                if int(cls_idx) == cidx:
                    want[cname] = True

    assert want["dog"], "Expected a dog with conf >= 0.80"
    assert want["bicycle"], "Expected a bicycle with conf >= 0.80"
    assert want["truck"], "Expected a truck with conf >= 0.80"
