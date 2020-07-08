#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from typing import Any, Dict


def replace(string: str, replace_dict: Dict[str, Any]) -> str:
    """Calls string.replace(k, v) for all key-value pairs in replace_dict (in order)."""
    for key, val in replace_dict.items():
        string = string.replace(key, str(val))
    return string
