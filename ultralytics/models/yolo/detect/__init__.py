# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .train_video_sequence import DetectionTrainerVS
from .val import DetectionValidator

__all__ = 'DetectionPredictor', 'DetectionTrainerVS', 'DetectionValidator'
