# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
import time
import warnings
from copy import copy

import numpy as np
import torch
from torch import distributed as dist


from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER, RANK, TQDM, colorstr
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first

class DetectionTrainerVS(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, augment=False, motion=self.args.motion, stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = False
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor. Not needed for classification but necessary for
        segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def train(self):
        """Train completed, evaluate and plot if specified by arguments."""
        world_size = 1

        # motion_pipeline = MotionPipeline("/home/lava/Documents/thermal_far/thermal_far/motion_module/motion_pipeline_config.yaml")

        self._setup_train(world_size)
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb),
                 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.run_callbacks('on_train_start')
        LOGGER.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n'
                    f'Using {self.train_loader.num_workers * (world_size or 1)} dataloader workers\n'
                    f"Logging results to {colorstr('bold', self.save_dir)}\n"
                    f'Starting training for {self.epochs} epochs...')
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.epochs  # predefine for resume fully trained model edge cases
        for epoch in range(self.start_epoch, self.epochs):
            self.epoch = epoch
            self.run_callbacks('on_train_epoch_start')
            self.model.train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                LOGGER.info('Closing dataloader mosaic')
                if hasattr(self.train_loader.dataset, 'mosaic'):
                    self.train_loader.dataset.mosaic = False
                if hasattr(self.train_loader.dataset, 'close_mosaic'):
                    self.train_loader.dataset.close_mosaic(hyp=self.args)
                self.train_loader.reset()

            if RANK in (-1, 0):
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            self.optimizer.zero_grad()
            for i, batch in pbar:
                self.run_callbacks('on_train_batch_start')
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round())
                    for j, x in enumerate(self.optimizer.param_groups):
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x['lr'] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x['initial_lr'] * self.lf(epoch)])
                        if 'momentum' in x:
                            x['momentum'] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                """
                # Motion module
                for j, img in enumerate(batch['img']):
                    # seq_id = (batch['im_file'][j].split('___')[-1]).split('.')[0]
                    img_numpy = img.detach().cpu().numpy().transpose((1, 2, 0))
                    input_new = motion_pipeline.run(img_numpy)
                    input_new = torch.Tensor(input_new).permute(2, 0, 1)
                    batch['img'][j] = input_new
                    # seq_id_before = seq_id
                """

                # Forward
                with torch.cuda.amp.autocast(self.amp):
                    batch = self.preprocess_batch(batch)
                    self.loss, self.loss_items = self.model(batch)
                    if RANK != -1:
                        self.loss *= world_size
                    self.tloss = (self.tloss * i + self.loss_items) / (i + 1) if self.tloss is not None \
                        else self.loss_items

                # Backward
                self.scaler.scale(self.loss).backward()

                # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()
                    last_opt_step = ni

                # Log
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
                loss_len = self.tloss.shape[0] if len(self.tloss.size()) else 1
                losses = self.tloss if loss_len > 1 else torch.unsqueeze(self.tloss, 0)
                if RANK in (-1, 0):
                    pbar.set_description(
                        ('%11s' * 2 + '%11.4g' * (2 + loss_len)) %
                        (f'{epoch + 1}/{self.epochs}', mem, *losses, batch['cls'].shape[0], batch['img'].shape[-1]))
                    self.run_callbacks('on_batch_end')
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks('on_train_batch_end')

            self.lr = {f'lr/pg{ir}': x['lr'] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()
            self.run_callbacks('on_train_epoch_end')

            if RANK in (-1, 0):

                # Validation
                self.ema.update_attr(self.model, include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
                final_epoch = (epoch + 1 == self.epochs) or self.stopper.possible_stop

                if self.args.val or final_epoch:
                    self.metrics, self.fitness = self.validate()
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                self.stop = self.stopper(epoch + 1, self.fitness)

                # Save model
                if self.args.save or (epoch + 1 == self.epochs):
                    self.save_model()
                    self.run_callbacks('on_model_save')

            tnow = time.time()
            self.epoch_time = tnow - self.epoch_time_start
            self.epoch_time_start = tnow
            self.run_callbacks('on_fit_epoch_end')
            torch.cuda.empty_cache()  # clears GPU vRAM at end of epoch, can help with out of memory errors

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                if RANK != 0:
                    self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks

        if RANK in (-1, 0):
            # Do final val with best.pt
            LOGGER.info(f'\n{epoch - self.start_epoch + 1} epochs completed in '
                        f'{(time.time() - self.train_time_start) / 3600:.3f} hours.')
            self.final_eval()
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks('on_train_end')
        torch.cuda.empty_cache()
        self.run_callbacks('teardown')

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)
