# Ultralytics YOLO 🚀, AGPL-3.0 license

from ultralytics.models.yolo import classify, detect, pose, segment, detect3d

from .model import YOLO

__all__ = 'classify', 'segment', 'detect', 'detect3d', 'pose', 'YOLO'
