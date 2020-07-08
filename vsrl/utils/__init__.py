#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from pathlib import Path

VSRL_WEBSITE = "http://safelearning.ai/vsrl/"

VSRL_PATH = Path.home() / ".vsrl"
VSRL_PATH.mkdir(exist_ok=True)
