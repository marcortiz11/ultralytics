# Ultralytics YOLO 🚀, AGPL-3.0 license
import cv2
import torch

from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops

from ultralytics.data.augment import multilabel_classify_augmentations


class MultilabelClassificationPredictor(BasePredictor):
    """
    A class extending the BasePredictor class for prediction based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.utils import ASSETS
        from ultralytics.models.yolo.classify import ClassificationPredictor

        args = dict(model='yolov8n-cls.pt', source=ASSETS)
        predictor = ClassificationPredictor(overrides=args)
        predictor.predict_cli()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = 'multilabel_classify'

    def preprocess(self, img):
        """Converts input image to model-compatible data type."""
        if not isinstance(img, torch.Tensor):
            self.transforms = multilabel_classify_augmentations(None, size=self.imgsz[0], augment=False, normalize_tensor=True)
            transformed_img = (self.transforms({'img': img[0], 'cls': []})['img'])
            img = torch.stack([self.transforms({'img': im, 'cls': []})['img'] for im in img], dim=0)
            """
            # Show image transformed
            import numpy as np
            np_image = transformed_img.permute((1, 2, 0)).detach().cpu().numpy() * 255
            cv2.imshow('Transformed image', np_image.astype(np.uint8))
            cv2.waitKey(5000)
            """
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        return img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32

    def postprocess(self, preds, img, orig_imgs):
        """Post-processes predictions to return Results objects."""
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=self.model.names, probs=pred))
        return results
