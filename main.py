import numpy as np
import pytorch_lightning
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
    DataStats,
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
    RandFlipd,
    DataStatsD,
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


# Function to initialize data lists
def init_data_lists():
    """
    This function initializes the image and mask paths.

    Args:
    base_data_path (Path): The base path of the data.

    Returns:
    list: A list of image paths.
    list: A list of mask paths.
    """
    base_data_path = Path(
        "/home/iejohnson/programing/Supervised_learning/DATA/preprocessed_data"
    )
    mask_paths = []
    image_paths = []
    for dir in base_data_path.iterdir():
        if dir.is_dir():

            image_paths.append(dir / f"{dir.name}_resampled_normalized_t2w.nii.gz")
            mask_paths.append(dir / f"{dir.name}_resampled_segmentations.nii.gz")

    return image_paths, mask_paths


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AddChannelD(object):
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            im = d[key]
            im = np.expand_dims(im, axis=0)
            d[key] = im
        return d


class Net(pytorch_lightning.LightningModule):
    """
    This class defines the network architecture.

    Attributes:
    _model (UNet): The UNet model.
    loss_function (DiceLoss): The loss function.
    post_pred (Compose): The post prediction transformations.
    post_label (Compose): The post label transformations.
    dice_metric (DiceMetric): The dice metric.
    best_val_dice (float): The best validation dice score.
    best_val_epoch (int): The epoch with the best validation dice score.
    validation_step_outputs (list): The outputs of the validation step.
    """

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
        self.loss_function = DiceLoss(softmax=True, to_onehot_y=True)
        self.post_pred = Compose(
            [EnsureType("tensor", device=device), AsDiscrete(to_onehot=5), DataStats()]
        )
        self.post_label = Compose(
            [EnsureType("tensor", device=device), AsDiscrete(to_onehot=5), DataStats()]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        self.validation_step_outputs = []
        self.prepare_data()

    def forward(self, x):
        """
        This function defines the forward pass of the network.

        Args:
        x (Tensor): The input tensor.

        Returns:
        Tensor: The output tensor.
        """
        return self._model(x)

    def prepare_data(self):
        """
        This function prepares the data for training and validation.
        """
        # set up the correct data path

        image_paths, mask_paths = init_data_lists()
        train_image_paths, test_image_paths, train_mask_paths, test_mask_paths = (
            train_test_split(image_paths, mask_paths, test_size=0.2)
        )
        # Prepare data
        train_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(train_image_paths, train_mask_paths)
        ]
        val_files = [
            {"image": img_path, "label": mask_path}
            for img_path, mask_path in zip(test_image_paths, test_mask_paths)
        ]

        train_files = train_files[:10]
        val_files = val_files[:1]
        # set the data transforms
        RandFlipd_prob = 0.35

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"]
                ),  # Add channel to image and mask so
                # Coarse Segmentation combine all mask
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=0),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=1),
                RandFlipd(keys=["image", "label"], prob=RandFlipd_prob, spatial_axis=2),
                # ToTensord(keys=["image", "label"], device=device),
                # DataStatsD(keys=["image", "label"]),
            ]
        )
        self.validation_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstD(
                    keys=["image", "label"]
                ),  # Add channel to image and mask so
                # ToTensord(keys=["image", "label"], device=device),
            ]
        )

        # we use cached datasets - these are 10x faster than regular datasets
        self.train_ds = CacheDataset(
            data=train_files,
            transform=self.train_transforms,
            cache_rate=0.5,
            num_workers=4,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=self.validation_transforms,
            cache_rate=0.5,
            num_workers=4,
        )

    def train_dataloader(self):
        """
        This function returns the training data loader.

        Returns:
        DataLoader: The training data loader.
        """
        train_loader = DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=4,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        """
        This function returns the validation data loader.

        Returns:
        DataLoader: The validation data loader.
        """
        val_loader = DataLoader(self.val_ds, batch_size=1, num_workers=4)
        return val_loader

    def configure_optimizers(self):
        """
        This function configures the optimizer.

        Returns:
        Adam: The Adam optimizer.
        """
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        This function defines the training step.

        Args:
        batch (dict): The batch of data.
        batch_idx (int): The index of the batch.

        Returns:
        dict: The loss and the logs.
        """
        images, labels = batch["image"], batch["label"]
        print("Training Step")
        print(f"Images.Shape = {images.shape}")
        print(f"Labels.Shape = {labels.shape}")
        output = self.forward(images)
        print(f"Output shape {output.shape}")
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        print("Validation Step")
        print(f"Images.Shape = {images.shape}")
        print(f"Labels.Shape = {labels.shape}")
        outputs = self.forward(images)
        print(f"Outputs shape {outputs.shape}")
        print(f"Shape before post_pred: {outputs.shape}")
        outputs = torch.stack(
            [self.post_pred(i) for i in decollate_batch(outputs)]
        )  # Stack the processed outputs back into a tensor
        labels = torch.stack(
            [self.post_label(i) for i in decollate_batch(labels)]
        )  # Do the same for labels if necessary
        print(f"Shape after post_pred: {outputs.shape}")
        loss = self.loss_function(outputs, labels)
        self.dice_metric(y_pred=outputs, y=labels)
        d = {"val_loss": loss, "val_number": len(outputs)}
        self.validation_step_outputs.append(d)
        return d

    def on_validation_epoch_end(self):
        """
        This function is called at the end of the validation epoch.
        """
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


if __name__ == "__main__":
    # initialise the LightningModule
    net = Net()

    # set up loggers and checkpoints
    log_dir = os.path.join(
        "/home/iejohnson/PycharmProjects/AML/AML_Project_Supervised", "logs"
    )
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir, name="lightning_logs"
    )
    os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        max_epochs=600,
        logger=tb_logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        num_sanity_val_steps=1,
        log_every_n_steps=16,
    )

    trainer.fit(net)
    print("Finished training")
