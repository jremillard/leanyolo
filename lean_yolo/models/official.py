from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class OfficialYOLOv10Adapter(nn.Module):
    def __init__(self, variant: str, nc: int = 80):
        super().__init__()
        import sys
        sys.path.insert(0, 'yolov10-official')
        from ultralytics.nn.tasks import YOLOv10DetectionModel

        yaml_path = f"ultralytics/cfg/models/v10/yolov10{variant}.yaml"
        self.off = YOLOv10DetectionModel(yaml_path, nc=nc)
        self.variant = variant

    def load_official_checkpoint(self, loaded_obj) -> None:
        # loaded_obj is an official checkpoint object with 'model' or similar inside
        sd = self.off.state_dict()
        # Try to use the underlying model object's state_dict if available
        if hasattr(loaded_obj, 'state_dict'):
            src = loaded_obj.state_dict()
        elif isinstance(loaded_obj, dict):
            model_obj = loaded_obj.get('model', None)
            src = model_obj.state_dict() if hasattr(model_obj, 'state_dict') else loaded_obj
        else:
            src = loaded_obj
        self.off.load_state_dict(src, strict=False)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Capture P3,P4,P5 from the official model graph using hooks
        feats = {}
        idxs = [16, 19, 22]
        hooks = []
        for i in idxs:
            hooks.append(self.off.model[i].register_forward_hook(lambda m, inp, out, i=i: feats.__setitem__(i, out)))
        with torch.no_grad():
            _ = self.off(x)
        for h in hooks:
            h.remove()
        p3, p4, p5 = feats[16], feats[19], feats[22]
        head = self.off.model[-1]
        # Produce training-format outputs per level
        return head.forward_feat([p3, p4, p5], head.cv2, head.cv3)

