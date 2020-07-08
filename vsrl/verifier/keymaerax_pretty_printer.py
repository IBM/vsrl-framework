#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from vsrl.verifier.expr import Expression, pp


def keymaerax_pp(e: Expression) -> str:
    return pp(e)
