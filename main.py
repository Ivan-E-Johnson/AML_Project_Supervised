import pytorch_lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, list_data_collate, decollate_batch, DataLoader
from monai.config import print_config
from monai.apps import download_and_extract
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import glob
from dotenv import load_dotenv
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImaged,
    EnsureChannelFirstD,
    SpatialResampleD,
    SpacingD,
    ResizeD,
    LoadImage,
    RandFlipD,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
    RandFlipd, DataStatsD,
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.transforms import (
    LoadImaged,
    ScaleIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
)
from sklearn.model_selection import train_test_split

print_config()
from pathlib import Path


def init_data_lists(base_data_path):
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():
            image_paths.append(dir / f"{dir.name}_prostate.nii.gz")
            mask_paths.append(dir / f"{dir.name}_segmentation.nii.gz")
    return image_paths, mask_paths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),  # Number of features in each layer
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(sigmoid=True)
        self.post_pred = Compose([EnsureType("tensor", device=device), AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([EnsureType("tensor", device=device), AsDiscrete(to_onehot=4)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []

        self.prepare_data()

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # set up the correct data path

        base_data_path = Path("/home/jsome/PycharmProjects/AML/DATA/SortedProstateData")
        image_paths, mask_paths = init_data_lists(base_data_path)
        train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = train_test_split(image_paths,
                                                                                                  mask_paths,
                                                                                                  test_size=0.2)
        # Prepare data
        train_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(train_image_paths, train_mask_paths)
        ]
        val_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(test_image_paths, test_mask_paths)
        ]

        train_files = train_files
        val_files = val_files
        # set the data transforms
        RandFlipd_prob = .35
        Spacing_dim = (1.5, 1.5, 3.0)
        Size_dim = (96, 96, 16)
        ScaleIntensity_Image = (0, 255)
        ScaleIntensity_Mask = (0, 1)
        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(keys=["image", "label"]),  # Add channel to image and mask so
                # DataStatsD(keys=["image", "label"]),
                SpacingD(keys=["image", "label"], pixdim=Spacing_dim, mode=("bilinear", "nearest")),
                # Downsample to 2mm spacing
                ResizeD(keys=["image", "label"], spatial_size=Size_dim, mode=("bilinear", "nearest")),
                ScaleIntensityd(keys=["image"], minv=ScaleIntensity_Image[0], maxv=ScaleIntensity_Image[1]),
                ScaleIntensityd(keys=["label"], minv=ScaleIntensity_Mask[0], maxv=ScaleIntensity_Mask[1]),
                # Coarse Segmentation combine all mask
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
                # DataStatsD(keys=["image", "label"]),

            ]
        )
        self.validation_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(keys=["image", "label"]),  # Add channel to image and mask so
                SpacingD(keys=["image", "label"], pixdim=Spacing_dim, mode=("bilinear", "nearest")),
                # Downsample to 2mm spacing
                ResizeD(keys=["image", "label"], spatial_size=Size_dim, mode=("bilinear", "nearest")),
                # DataStatsD(keys=["image", "label"]),
                ScaleIntensityd(keys=["image"], minv=ScaleIntensity_Image[0], maxv=ScaleIntensity_Image[1]),
                ScaleIntensityd(keys=["label"], minv=ScaleIntensity_Mask[0], maxv=ScaleIntensity_Mask[1]),
                # Coarse Segmentation combine all mask
                # DataStatsD(keys=["image", "label"]),

            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=1.0,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.validation_transforms,
            cache_rate=1.0,
            num_workers=4,
        )

    #         self.train_ds = monai.data.Dataset(
    #             data=train_files, transform=train_transforms)
    #         self.val_ds = monai.data.Dataset(
    #             data=val_files, transform=val_transforms)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        print(output.shape)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        print(images.shape)
        print(labels.shape)
        roi_size = (96, 96, 24)
        # outputs = sliding_window_inference(images,roi_size,1, self.forward)
        outputs = self.forward(images)
        print(outputs.shape)
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        mean_val_dice = self.dice_metric.aggregate().item()
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.log("val_dice", mean_val_dice, on_step=False, on_epoch=True)
        return d

    def on_validation_epoch_end(self):
        val_loss, num_items = 0, 0
        for output in self.validation_step_outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.validation_step_outputs.clear()  # free memory
        return {"log": tensorboard_logs}


class BestModelCheckpoint(pytorch_lightning.callbacks.Callback):
    def __init__(self, monitor='val_dice', mode='max'):
        super().__init__()
        self.monitor = monitor
        self.mode = mode

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        if logs is not None:
            val_dice = logs.get(self.monitor)
            if val_dice is not None:
                if self.mode == 'max' and val_dice >= pl_module.best_val_dice:
                    pl_module.best_val_dice = val_dice
                    pl_module.best_val_epoch = trainer.current_epoch
                    # Save the best model
                    checkpoint_callback = trainer.checkpoint_callback  # Access checkpoint callback from trainer
                    checkpoint_callback.best_model_path = os.path.join(checkpoint_callback.dirpath, 'best_model.pth')
                    trainer.save_checkpoint(checkpoint_callback.best_model_path)


# initialise the LightningModule
net = Net()

# set up loggers and checkpoints
log_dir = os.path.join("/home/jsome/PycharmProjects/AML/AML_Project_Supervised", "logs")
tb_logger = pytorch_lightning.loggers.TensorBoardLogger(save_dir=log_dir)

checkpoint_callback = ModelCheckpoint(
    monitor='val_dice',
    mode='max',
    save_last=True,
    dirpath=log_dir,
    filename='checkpoint-{epoch:02d}-{val_dice:.2f}',
)
# Initialise Lightning's trainer with the custom callback
trainer = pytorch_lightning.Trainer(
    max_epochs=600,
    logger=tb_logger,
    enable_checkpointing=True,
    num_sanity_val_steps=1,
    log_every_n_steps=16,
    callbacks=[BestModelCheckpoint(), checkpoint_callback],  # Add the custom callback
)

trainer.fit(net)
