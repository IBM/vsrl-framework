#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from pathlib import Path
from typing import Optional, Sequence

import pytorch_lightning as pl
from comet_ml.config import get_api_key, get_config

from ..symmap.detectors.center_track import CenterTrack
from ..utils.logger import CometLogger


def train_center_track(
    config_path: str,
    save_dir: str,
    model_type: str = "resnet_small",
    devices: Optional[Sequence[int]] = None,
    max_epochs: int = 2_000,
    epoch_size: int = 5_000,
    batch_size: int = 32,
    n_workers: int = 6,
    img_scale: int = 1,
    label_scale: int = 4,
    grayscale: bool = False,
    prog_bar: bool = True,
    force_offline: bool = False,
    upload_model: bool = False,
    use_logger: bool = True,
):
    """
    Train a CenterTrack / CenterNet object detector.

    :param config_path: path to TOML file listing images used for training
      (see assets/configs or the README for examples)
    :param save_dir: directory where the trained model should be saved
    :param model_type: one of {resnet, resnet_small}. resnet is a ResNet18; the small
      variant only keeps the first residual block.
    :param devices: index of the GPUs to use for training
    :param max_epochs: the maximum number of training epochs
    :param epoch_size: number of batches per training epoch
    :param batch_size: number of images per batch
    :param n_workers: number of worker processes to generate training data
    :param img_scale: factor by which to downscale the images before generating training
      data. E.g. `img_scale=2` means the images will be half the original size.
    :param label_scale: factor by which the labels are smaller than the images. This
      applies after the images are downscaled by `img_scale`.
    :param grayscale: whether to convert the images to grayscale
    :param prog_bar: whether to display a progress bar for the training
    :param force_offline: if True, no data will be sent to comet.ml even if an API key
      is found (data can only be sent if an API key is found, e.g. in ~/.comet.config)
    :param upload_model: whether to upload the final checkpoint to comet.ml
    :param use_logger: if True, a CometLogger is used; otherwise, no logging is done
    """
    config_path = Path(config_path)

    model = CenterTrack(
        model=model_type,
        config_path=config_path,
        epoch_size=epoch_size,
        batch_size=batch_size,
        n_workers=n_workers,
        img_scale=img_scale,
        label_scale=label_scale,
        grayscale=grayscale,
    )
    print(f"# parameters: {sum(p.numel() for p in model.parameters()):,}")

    api_key = get_api_key(None, get_config())
    project_name = "vsrl"
    exp_name = f"detector_{config_path.name.split('.')[0]}"
    if use_logger:
        logger = CometLogger(
            api_key,
            save_dir,
            project_name=project_name,
            experiment_name=exp_name,
            force_offline=force_offline,
            log_env_gpu=False,
            log_env_cpu=False,
        )

        logger.log_hyperparams(
            {
                "img_scale": img_scale,
                "grayscale": grayscale,
                "model_type": model_type,
                "label_scale": label_scale,
                "config_name": config_path.name,
            }
        )
        checkpoint_callback = True
    else:
        logger = False
        checkpoint_callback = False
        upload_model = False

    trainer = pl.Trainer(
        logger,
        gpus=devices,
        max_epochs=max_epochs,
        progress_bar_refresh_rate=int(prog_bar),
        checkpoint_callback=checkpoint_callback,
    )
    trainer.fit(model)
    if upload_model:
        save_dir = Path(logger.save_dir) / logger.name / logger.version / "checkpoints"
        files = list(save_dir.glob("*"))
        if len(files) > 1:
            print("Warning: found multiple checkpoint files; only uploading one.")
        logger.experiment.log_asset(files[-1])
    return model
