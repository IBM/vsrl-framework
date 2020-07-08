#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

# TODO add version checking so that when assets are updated we know that it's time to download again.
import logging
import shutil
import urllib
import zipfile

from vsrl.utils import VSRL_PATH, VSRL_WEBSITE

IMAGES_DIRECTORY = VSRL_PATH / "images"


def get_image_path(filename: str) -> str:
    """
    Gets the location of an image as a string. If the images directory does not exist,
    then assets are downloaded from safelearning.ai
    :param filename: The filename to retrieve.
    :return: The location of the image.
    """
    image_location = IMAGES_DIRECTORY / filename
    if not image_location.exists():
        logging.info(
            f"Could not find {filename} in the assets directory, so we're re-downloading the assets."
        )
        _download_assets(True)
    assert (
        image_location.exists()
    ), f"Could not find asset ${filename} in location ${image_location}."
    return str(image_location)


def _download_assets(force_reload: bool = False) -> None:
    img_dir = IMAGES_DIRECTORY
    if force_reload and img_dir.exists():
        shutil.rmtree(img_dir)
    if not img_dir.exists():
        img_dir.mkdir(exist_ok=True, parents=True)
        data = urllib.request.urlopen(VSRL_WEBSITE + "/assets/assets.zip").read()
        assets_zip_loc = img_dir / "assets.zip"
        assets_zip_loc.write_bytes(data)
        assert zipfile.is_zipfile(assets_zip_loc)
        zipfile.ZipFile(assets_zip_loc).extractall(img_dir)
        assets_zip_loc.unlink()
