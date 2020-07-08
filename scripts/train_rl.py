#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from auto_argparse import parse_args_and_run

from vsrl.training.train_rl import run

if __name__ == "__main__":
    parse_args_and_run(run)
