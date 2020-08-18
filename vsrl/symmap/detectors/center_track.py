#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Note - the detector here is actually a CenterNet model, not CenterTrack;
tracking is planning to be added in a future version.
"""

import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from ..data import ImgDataset, parse_config
from .utils import fast_focal_loss, offset_loss


# from CenterTrack: note that they set bias=False except for in the output heads
def conv3x3(
    in_channels: int, out_channels: int, stride: int = 1, bias: bool = False
) -> nn.Module:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias
    )


class UpConv(nn.Module):
    # NB: CenterNet does (DCN, BN, ReLU, CT, BN, ReLU) each time (DCN is deform conv, CT
    # is conv transpose) - DCN changes channels, CT changes resolution - but CenterTrack
    # just does a ConvTranspose to change both. The ResNet code from CenterTrack looks
    # broken, though...
    #     return nn.Sequential(
    #         conv3x3(in_channels, out_channels),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace=True),
    #         nn.ConvTranspose2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(inplace=True),
    #     )
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, output_size=None):
        x = self.upconv(x, output_size=output_size)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResNetCT(nn.Module):
    def __init__(
        self, n_classes: int, pretrained=True, freeze=False, grayscale: bool = False,
    ):
        assert False, "Update normalization"
        super().__init__()
        self.n_classes = n_classes
        resnet18 = torchvision.models.resnet18(pretrained=pretrained)

        if grayscale:
            self.normalize_mean = torch.tensor([0.449]).reshape(1, -1, 1, 1)
            self.normalize_std = torch.tensor([0.226]).reshape(1, -1, 1, 1)
            # only difference is 1 input channel instead of 3
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
                1, -1, 1, 1
            )
            self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).reshape(
                1, -1, 1, 1
            )
            self.conv1 = resnet18.conv1

        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4

        if freeze:
            self.conv1.weight.requires_grad = False
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.upconv1 = UpConv(512, 256)
        self.upconv2 = UpConv(256, 128)
        self.upconv3 = UpConv(128, 64)

        # the first conv layers could possibly be omitted
        self.keypoint_classifier = nn.Sequential(
            conv3x3(64, 256, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_classes, 1),
            nn.Sigmoid(),
        )
        self.offset_predictor = nn.Sequential(
            conv3x3(64, 256, bias=True), nn.ReLU(inplace=True), nn.Conv2d(256, 2, 1),
        )

    def forward(self, x):
        # normal resnet18

        ## stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ## res blocks
        x = self.layer1(x)
        upconv3_size = x.shape[2:]
        x = self.layer2(x)
        upconv2_size = x.shape[2:]
        x = self.layer3(x)
        upconv1_size = x.shape[2:]
        x_small = self.layer4(x)

        # added upsampling layers
        x = self.upconv1(x_small, output_size=upconv1_size)
        x = self.upconv2(x, output_size=upconv2_size)
        x = self.upconv3(x, output_size=upconv3_size)

        output_heads = {
            "probs": self.keypoint_classifier(x),
            "offset": self.offset_predictor(x),
        }

        return output_heads


class ResNetCT0UC_SH(nn.Module):
    def __init__(
        self, n_classes: int, pretrained=True, freeze=False, grayscale: bool = False,
    ):
        super().__init__()
        self.n_classes = n_classes
        resnet18 = torchvision.models.resnet18(pretrained=pretrained)

        if grayscale:
            normalize_mean = torch.tensor([0])
            normalize_std = torch.tensor([1])
            # only difference is 1 input channel instead of 3
            self.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            normalize_mean = torch.tensor([0] * 3)
            normalize_std = torch.tensor([1] * 3)
            self.conv1 = resnet18.conv1
        self._normalize_mean = nn.Parameter(
            normalize_mean.float().reshape(1, -1, 1, 1), requires_grad=False
        )
        self._normalize_std = nn.Parameter(
            normalize_std.float().reshape(1, -1, 1, 1), requires_grad=False
        )

        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1

        if freeze:
            self.conv1.weight.requires_grad = False
            self.bn1.weight.requires_grad = False
            self.bn1.bias.requires_grad = False
            for param in self.layer1.parameters():
                param.requires_grad = False

        self.keypoint_classifier = nn.Sequential(
            nn.Conv2d(64, n_classes, 1), nn.Sigmoid(),
        )
        self.offset_predictor = nn.Conv2d(64, 2, 1)

    def forward(self, x):
        # normal resnet18

        x = (x - self._normalize_mean) / self._normalize_std

        ## stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        ## res blocks
        x = self.layer1(x)

        output_heads = {
            "probs": self.keypoint_classifier(x),
            "offset": self.offset_predictor(x),
        }
        return output_heads


class CenterTrack(pl.LightningModule):
    def __init__(
        self,
        model: str,
        config_path: str = "",
        n_classes: Optional[int] = None,
        epoch_size: int = 5_000,
        batch_size: int = 32,
        lambda_offset: float = 1.0,
        n_workers: int = 4,
        grayscale: bool = False,
        img_scale: int = 1,
        label_scale: int = 4,
        fit_mean_std: bool = True,
    ):
        """
        :param pretrained: whether to use pretrained resnet18 or train it from scratch
        :param freeze: whether to freeze the weights in resnet18
        :param fit_mean_std: if True, fit channel-wise mean and standard deviation
          parameters to one epoch's worth of generated images. Otherwise, the default
          means and standard deviations from the model will be kept.
        """
        super().__init__()
        if config_path:
            config_path = Path(config_path).resolve()
            config_path = str(config_path)
            try:
                n_classes = len(parse_config(config_path)["objects"])
            except FileNotFoundError:
                msg = (
                    f"Config file not found at {config_path}. "
                    "Correct the path if you need to generate data for training."
                )
                if n_classes is None:
                    msg2 = (
                        " If you only want to use a (trained) model, you can give "
                        "`n_classes` instead of `config_path`."
                    )
                    raise ValueError(msg + msg2)
                warnings.warn(msg)
        elif n_classes is None:
            raise ValueError("n_classes must be given if config_path is not.")

        self.save_hyperparameters()
        try:
            self.model = MODELS[model](n_classes, grayscale=grayscale)
        except KeyError:
            raise ValueError(
                f"Model {model} not recognized; valid models are {list(MODELS.keys())}."
            )
        self.n_classes = n_classes
        self.config_path = config_path
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.lambda_offset = lambda_offset
        self.n_workers = n_workers
        self.grayscale = grayscale
        self.img_scale = img_scale
        self.label_scale = label_scale
        self._fit_mean_std = fit_mean_std
        self.step = 0
        self.dset = None

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        try:
            config = parse_config(self.config_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found at {self.config_path}.")

        bg_imgs = [Image.open(path) for path in config["backgrounds"]]
        obj_imgs = {
            name: [Image.open(path) for path in paths]
            for name, paths in config["objects"].items()
        }

        # load all images now instead of using lazy-loading; this plays more nicely with
        # multiple data loader workers
        for img in bg_imgs:
            img.load()
        for img_list in obj_imgs.values():
            for img in img_list:
                img.load()

        self.dset = ImgDataset(
            bg_imgs,
            obj_imgs,
            self.epoch_size,
            grayscale=self.grayscale,
            img_scale=self.img_scale,
            label_scale=self.label_scale,
        )

        if self._fit_mean_std:
            imgs = []
            for _ in range(self.epoch_size):
                imgs.append(self.dset[0][0])

            imgs = torch.stack(imgs)
            means = imgs.mean([0, 2, 3])
            stds = (imgs - means).std([0, 2, 3])
            self.model._normalize_mean[:] = means
            self.model._normalize_std[:] = stds

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.n_workers,
        )

    def configure_optimizers(self):
        # lr from CenterTrack default for batch size 32
        optimizer = torch.optim.Adam(self.parameters(), lr=1.25e-4)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        imgs, labels = batch
        preds = self(imgs)
        is_center = labels["probs"] == 1
        n_objects = is_center.sum()
        loss_focal = fast_focal_loss(
            preds["probs"], labels["probs"], is_center, n_objects
        )
        loss_offset = offset_loss(
            preds["offset"], labels["offset"], is_center, n_objects
        )

        loss = loss_focal + self.lambda_offset * loss_offset

        if self.step % 50 == 0 and self.logger:
            metrics = {
                "loss": loss,
                "focal_loss": loss_focal,
                "offset_loss": loss_offset,
            }
            self.logger.log_metrics(
                metrics, step=self.step,
            )
        self.step += 1
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        imgs, labels = batch
        preds = self(imgs)
        is_center = labels["probs"] == 1
        n_objects = is_center.sum()
        loss_focal = fast_focal_loss(
            preds["probs"], labels["probs"], is_center, n_objects
        )
        loss_offset = offset_loss(
            preds["offset"], labels["offset"], is_center, n_objects
        )
        loss = loss_focal + self.lambda_offset * loss_offset

        losses = {
            "val_focal_loss": loss_focal,
            "val_offset_loss": loss_offset,
            "val_loss": loss,
        }

        return losses

    def validation_epoch_end(self, outputs):
        keys = outputs[0].keys()
        avg_metrics = {}
        for key in keys:
            avg_metrics[key] = torch.stack([o[key] for o in outputs]).float().mean()
        avg_metrics["lr"] = self.trainer.optimizers[0].param_groups[0]["lr"]
        if self.logger:
            self.logger.log_metrics(avg_metrics, step=self.step)
        return avg_metrics

    def detect(
        self, imgs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param img: NCHW uint8 tensor on the same device as `self`.
        :returns: img_idx, class_idx, center_x, center_y
        """
        # See https://github.com/astooke/rlpyt/blob/668290d1ca94e9d193388a599d4f719bc3a23fba/rlpyt/models/pg/atari_ff_model.py#L40
        # if there are shape issues (e.g. imgs is MNHWC instead of NHWC)

        imgs = imgs.type(torch.float32)
        imgs.mul_(1.0 / 255)

        with torch.no_grad():
            preds = self(imgs)
        probs = preds["probs"]
        offset = preds["offset"]
        local_max = torch.nn.functional.max_pool2d(probs, 3, stride=1, padding=1)
        detect_idx = ((probs > 0.5) & (probs == local_max)).nonzero(as_tuple=True)
        img_idx, class_idx, pix_y, pix_x = detect_idx
        offset_x, offset_y = offset[img_idx, :, pix_y, pix_x].transpose(0, 1)
        center_x = self.label_scale * (pix_x + offset_x)
        center_y = self.label_scale * (pix_y + offset_y)
        return img_idx, class_idx, center_x, center_y, probs[detect_idx]


MODELS = {"resnet": ResNetCT, "resnet_small": ResNetCT0UC_SH}
