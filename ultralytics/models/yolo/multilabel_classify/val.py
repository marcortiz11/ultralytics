# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.data import MultilabelClassificationDataset, build_dataloader
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import MultilabelClassifyMetrics, ConfusionMatrix
from ultralytics.utils.plotting import plot_images


class MultilabelClassificationValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics.models.yolo.classify import ClassificationValidator

        args = dict(model='yolov8n-cls.pt', data='imagenet10')
        validator = ClassificationValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initializes ClassificationValidator instance with args, dataloader, save_dir, and progress bar."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.targets = None
        self.pred = None
        self.args.task = 'multilabel_classify'
        self.metrics = MultilabelClassifyMetrics()

    def get_desc(self):
        """Returns a formatted string summarizing classification metrics."""
        return ('%22s' + '%11s'*3) % ('classes', 'hamming', 'precision', 'recall')

    def init_metrics(self, model):
        """Initialize confusion matrix, class names, and top-1 and top-5 accuracy."""
        self.names = model.names
        self.nc = len(model.names)
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf, task='multilabel_classify')
        self.pred = []
        self.targets = []

    def preprocess(self, batch):
        """Preprocesses input batch and returns it."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True)
        batch['img'] = batch['img'].half() if self.args.half else batch['img'].float()
        batch['cls'] = batch['cls'].to(self.device)
        return batch

    def update_metrics(self, preds, batch):
        """Updates running metrics with model predictions and batch targets."""
        self.pred.append((preds > 0.5) * 1.0)
        batch['cls'] = torch.round(batch['cls'])
        self.targets.append(batch['cls'])

    def finalize_metrics(self, *args, **kwargs):
        """Finalizes metrics of the model such as confusion_matrix and speed."""
        """
        self.confusion_matrix.process_cls_preds(self.pred, self.targets)
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir,
                                           names=self.names.values(),
                                           normalize=normalize,
                                           on_plot=self.on_plot)
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def get_stats(self):
        """Returns a dictionary of metrics obtained by processing targets and predictions."""
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def build_dataset(self, img_path):
        return MultilabelClassificationDataset(root=img_path, args=self.args, augment=False, prefix='val')

    def get_dataloader(self, dataset_path, batch_size):
        """Builds and returns a data loader for classification tasks with given parameters."""
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def print_results(self):
        """Prints evaluation metrics for YOLO object detection model."""
        pf = '%22s' + '%11.3g' * 3  # print format
        LOGGER.info(pf % ('all', self.metrics.hamming, self.metrics.precision, self.metrics.recall))

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            images=batch['img'],
            batch_idx=torch.arange(len(batch['img'])),
            cls=batch['cls'],  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f'val_batch{ni}_labels.jpg',
            names=self.names,
            on_plot=self.on_plot)

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(batch['img'],
                    batch_idx=torch.arange(len(batch['img'])),
                    cls=torch.round(preds),
                    fname=self.save_dir / f'val_batch{ni}_pred.jpg',
                    names=self.names,
                    on_plot=self.on_plot)  # pred
