#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import torch


class SymFeatExtractor(torch.nn.Module):
    """
    This class is not intended for use by itself; each environment should subclass it to
    provide the proper conversion from detections to symbolic features.

    Any symbolic features for which the ground-truth from the environment should be used
    should have a nan.
    """

    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, imgs):
        self.detector.eval()
        with torch.no_grad():
            img_idx, obj_id, center_x, center_y, probs = self.detector.detect(imgs)
        return img_idx, obj_id, center_x, center_y, probs
