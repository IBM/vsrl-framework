#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import os
from distutils.core import setup

from setuptools import find_packages

opencv_pkg = ""
if "DISPLAY" not in os.environ.keys():
    opencv_pkg = "opencv-python-headless"
else:
    opencv_pkg = "opencv-python"

setup(
    name="vsrl",
    version="0.0.1",
    description="Visceral: A Framework for Verifiably Safe Reinforcement Learning",
    author="IBM Research",
    author_email="visceral@safelearning.ai",
    url="https://visceral.safelearning.ai",
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "torch",
        "pillow",
        opencv_pkg,
        "pytorch_lightning",
        "comet_ml",
        "psutil",
        "torchvision",
        "parsimonious",
        "matplotlib",
        "portion",
        "toml",
        "auto-argparse",
        "rlpyt @ git+https://github.com/astooke/rlpyt.git",
        "gym",
    ],
    extras_require={"dev": ["pytest", "pytest-cov"]},
)
