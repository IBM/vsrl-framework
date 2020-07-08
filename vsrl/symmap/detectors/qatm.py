#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Original copied from https://github.com/kamata1729/QATM_pytorch on 7/12/19;
license below. Modifications copyright 2019, Nathan Hunt, MIT license.

MIT License

Copyright (c) 2019 Hiromichi Kamata

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms

from .utils import focal_loss


class FeatureExtractor(nn.Module):
    """
    This wraps a (pretrained) model and extracts some of its features.

    The features from two different layers of the model are saved (no
    unnecessary layers are used at all if the model is of Sequential type) and then one
    is interpolated to the size of the other. All parameters of the model are set not to
    require gradients to speed up the computation, so no further training is expected.

    Suggestions
    * Using the output of a conv layer is probably better than a ReLU layer
    * Using one layer from near the start of the network and one from the (late) middle may
      provide good high- and low-level features (layers that are too late in the network
      may be too specialized for the pretrained task).
    * Also consider the amount of pooling done before the layers you use; this should be
      reasonable given the size of your templates.
    """

    def __init__(
        self,
        model: nn.Module,
        layer1_idx: Union[str, float],
        layer2_idx: Union[str, float],
        device: Union[str, torch.device] = "cpu",
        trainable: bool = False,
        half: bool = False,
        extra_feature_conv: bool = False,
        conv_interpolate: bool = False,
    ):
        """
        :param layer1_idx: If an int, the layer is retrieved by indexing into the model;
          this requires a Sequential type network. If a float, it is converted to an int
          as `layer1_idx = int(layer1_idx * n_layers)`. If a string, the layer is retrieved
          as `getattr(model, layer1_idx)`.
        :param layer2_idx: as layer1_idx except assumed to occur later in the network and
          output features with height / width not larger than layer1. layer2_idx cannot
          be a string unless layer1_idx is too.
        :param trainable: if not True, all parameters are set to not require gradients.
        """
        super().__init__()

        if isinstance(layer1_idx, float):
            layer1_idx = int(layer1_idx * len(model))
        if isinstance(layer2_idx, float):
            layer2_idx = int(layer2_idx * len(model))

        self.layer1_idx = layer1_idx
        self.layer2_idx = layer2_idx
        self.device = device
        self.feature1 = None  # type: torch.Tensor
        self.feature2 = None  # type: torch.Tensor

        if isinstance(layer1_idx, str):
            self.model = model
            layer1 = getattr(self.model, layer1_idx)
            layer2 = getattr(self.model, layer2_idx)
        else:
            # save some time by not computing unnecessary layers
            self.model = model[: layer2_idx + 1]
            layer1 = self.model[layer1_idx]
            layer2 = self.model[layer2_idx]

        layer1.register_forward_hook(self.save_feature1)
        layer2.register_forward_hook(self.save_feature2)
        self.layers = [layer1, layer2]

        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

        # https://pytorch.org/docs/stable/torchvision/models.html
        tforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if half:
            tforms.append(lambda x: x.half())
        self.transform = transforms.Compose(tforms)

        self.extra_feature_conv = extra_feature_conv
        if extra_feature_conv:
            in_channels = sum(layer.out_channels for layer in self.layers)
            self.conv = torch.nn.Conv2d(in_channels, in_channels, 3, padding=1)
            self.non_linearity = torch.nn.ReLU(inplace=True)

    def save_feature1(
        self, module: nn.Module, inputs: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self.feature1 = output

    def save_feature2(
        self, module: nn.Module, inputs: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        self.feature2 = output

    def __call__(self, inputs: torch.Tensor, mode: str = "big") -> torch.Tensor:
        """
        :param inputs: (batch_size x channels x height x width), RGB order, dtype float32
          This tensor should already have been normalized as self.transform does.
        """
        self.model(inputs)

        if mode == "big":
            # resize feature1 to the same size as feature2
            self.feature1 = F.interpolate(
                self.feature1,
                size=self.feature2.shape[2:4],
                mode="bilinear",
                align_corners=True,
            )
        else:
            # resize feature2 to the same size as feature1
            self.feature2 = F.interpolate(
                self.feature2,
                size=self.feature1.shape[2:4],
                mode="bilinear",
                align_corners=True,
            )

        features = torch.cat((self.feature1, self.feature2), dim=1)
        if self.extra_feature_conv:
            return self.conv(self.non_linearity(features))
        return features


class QATM(nn.Module):
    """
    * templates: Dict[str, np.ndarray] where the array is (height x width x channels),
      RGB order, dtype uint8
    * template_features: Dict[str, torch.Tensor] where the tensor is
      (1 x channels x height x width), RGB order, dtype uint8
    * template_layers: Dict[str, Tuple[nn.Module, nn.Module]] where the modules are
      (padding_layer, smoothing_layer)
    """

    def __init__(
        self,
        alpha: float = 25.0,
        model: Optional[nn.Module] = None,
        layer1_idx: Union[str, float] = 2,
        layer2_idx: Union[str, float] = 16,
        device: Union[torch.device, str] = "cpu",
        train_feature_extractor: bool = False,
        train_smoothing_layers: bool = True,
        train_alpha: bool = True,
        mean_std_norm: str = "combined",
        positive_func: str = "softmax",
        max_norm: bool = False,
        batchnorm: bool = False,
        l2: bool = True,
        max_sum: bool = False,
        half: bool = False,
        extra_feature_conv: bool = False,
        conv_interpolate: bool = False,
        tracker=None,
    ):
        """
        :param mean_std_norm: how to normalize the template and image features
          * "combined": use the combined mean and std of each (image, template) pair
          * "individual": normalize each image and each template individually
          * "": apply no mean-std normalization
        :positive_func: {softmax, exp, sigmoid}
        """
        super().__init__()
        self.template_shapes: Dict[str, Tuple[int, int]] = {}

        if model is None:
            model = models.vgg19(pretrained=True).features

        self.alpha_img = nn.Parameter(
            torch.tensor(float(alpha)), requires_grad=train_alpha
        )
        self.alpha_template = nn.Parameter(
            torch.tensor(float(alpha)), requires_grad=train_alpha
        )
        self.feature_extractor = FeatureExtractor(
            model,
            layer1_idx,
            layer2_idx,
            device,
            train_feature_extractor,
            half,
            extra_feature_conv,
            conv_interpolate,
        )
        self.tracker = tracker
        self.device = device
        self.train_smoothing_layers = train_smoothing_layers
        self.mean_std_norm = mean_std_norm
        self.positive_func = positive_func
        self.templates = {}  # type: Dict[str, np.ndarray]
        self.template_features = {}  # type: Dict[str, torch.Tensor]
        self.template_layers = {}  # type: Dict[str, Tuple[nn.Module, nn.Module]]
        self.threshes = {}  # type: Dict[str, float]
        self._img_features = None  # type: Optional[torch.Tensor]
        self.stacked_threshes = None  # type: Optional[torch.Tensor]
        self._idx = None  # type: Optional[torch.Tensor]
        self._mean = nn.Parameter(
            torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, -1, 1, 1),
            requires_grad=False,
        )
        self._std = nn.Parameter(
            torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, -1, 1, 1),
            requires_grad=False,
        )
        self.to(self.device)
        self.is_half = half
        self.eval()
        if half:
            self.half()

        self.max_norm = max_norm
        self.batchnorm = batchnorm
        self.l2 = l2
        self.max_sum = max_sum
        self.extra_feature_conv = extra_feature_conv
        self.conv_interpolate = conv_interpolate

        if conv_interpolate:
            channels = 1
            self.conv_transpose1 = torch.nn.ConvTranspose2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                stride=2,
                output_padding=[1, 0],
            )
            self.conv_transpose2 = torch.nn.ConvTranspose2d(
                channels, channels, kernel_size=3, stride=2, output_padding=[1, 1]
            )
            initialize_bilinear_interpolation(self.conv_transpose1)
            initialize_bilinear_interpolation(self.conv_transpose2)

        if self.batchnorm:
            self.bn_layer = torch.nn.BatchNorm2d(320)

    def register_template(self, template: np.ndarray, name: str, thresh: float) -> None:
        """
        :param template: (height x width x channels) in RGB order, dtype of uint8
        """
        self.templates[name] = template
        self.template_shapes[name] = tuple(template.shape[:-1])
        self.threshes[name] = thresh
        # maintain sorted order so that we get the templates out in the same order as
        # the labels are stacked together
        self.templates = {k: self.templates[k] for k in sorted(self.templates.keys())}
        self.stacked_threshes = nn.Parameter(
            torch.tensor(
                [self.threshes[k] for k in self.templates], device=self.device
            ).reshape(1, -1, 1, 1),
            requires_grad=False,
        )
        template_features = self.feature_extractor(
            self.feature_extractor.transform(template).unsqueeze(0).to(self.device)
        )
        # add as a parameter so that the features stay on the right device
        setattr(
            self,
            f"_template_features_{name}",
            nn.Parameter(template_features, requires_grad=False),
        )
        self.template_features[name] = getattr(self, f"_template_features_{name}")

        height, width = template.shape[:2]
        pad_left = width // 2
        pad_right = width - 1 - pad_left
        pad_top = height // 2
        pad_bottom = height - 1 - pad_top
        padding_layer = nn.ZeroPad2d([pad_left, pad_right, pad_top, pad_bottom])
        self.add_module(f"{name}_pad", padding_layer)

        smoothing_layer = nn.Conv2d(1, 1, [height, width], bias=True)
        smoothing_layer.weight.data = torch.ones_like(smoothing_layer.weight.data)
        smoothing_layer.weight.requires_grad = self.train_smoothing_layers
        self.add_module(f"{name}_conv", smoothing_layer)

        self.template_layers[name] = (padding_layer, smoothing_layer)
        self.to(self.device)
        if self.is_half:
            self.half()

    def __call__(
        self, imgs: torch.Tensor, template_name: str, reset_img_features: bool = True
    ) -> torch.Tensor:
        """
        :param imgs: (n_imgs x channels x height x width), RGB order, dtype float32.
          These should already be normalized as feature_extractor.transform does.
        :returns: scores (n_imgs x height x width)
        """
        height, width = self.templates[template_name].shape[:2]
        padding_layer, smoothing_layer = self.template_layers[template_name]

        if reset_img_features or self._img_features is None:
            self._img_features = self.feature_extractor(imgs)
        img_features = self._img_features

        if self.extra_feature_conv:
            template = self.templates[template_name]
            template_features = self.feature_extractor(
                self.feature_extractor.transform(template).unsqueeze(0).to(self.device)
            )
        else:
            template_features = self.template_features[template_name]
        template_features = template_features.repeat(len(self._img_features), 1, 1, 1)

        if self.mean_std_norm == "combined":
            img_features, template_features = self._norm(
                img_features, template_features
            )
        elif self.mean_std_norm == "individual":
            img_features = self._norm(img_features)
            template_features = self._norm(template_features)

        if self.batchnorm:
            img_features, template_features = self.do_bn(
                img_features, template_features
            )

        if self.max_norm:
            img_norm = torch.norm(img_features, dim=1, keepdim=True)
            if getattr(self, "img_norm_max", None) is None:
                self.img_norm_max = img_norm.max(0, keepdim=True).values
            else:
                self.img_norm_max = (
                    torch.cat((img_norm, self.img_norm_max)).max(0, keepdim=True).values
                )
            self.img_norm_max = self.img_norm_max.detach()

            temp_norm_max = getattr(self, f"{template_name}_norm_max", None)
            if temp_norm_max is None:
                temp_norm_max = (
                    torch.norm(template_features, dim=1, keepdim=True)
                    .max(0, keepdim=True)
                    .values
                )
                setattr(self, f"{template_name}_norm_max", temp_norm_max.detach())

            img_features = img_features / self.img_norm_max
            template_features = template_features / temp_norm_max
        elif self.l2:
            img_features = F.normalize(img_features)
            template_features = F.normalize(template_features)

        dist = torch.einsum("nchw,ncyx->nhwyx", img_features, template_features)
        score = self._qatm(dist, template_name)
        score = score.log()
        # batch x height x width x channels -> batch x channels x height x width
        # see if moving the [..., 1] in self._qatm removes need for transpose here
        score = score.transpose(3, 1).transpose(2, 3)
        if self.conv_interpolate:
            score = self.conv_transpose2(self.conv_transpose1(score))
        else:
            score = F.interpolate(
                score, imgs.shape[-2:], mode="bilinear", align_corners=True
            )
        score = smoothing_layer(padding_layer(score))

        score[..., :, : width // 2] = 0
        score[..., :, math.ceil(-width / 2) :] = 0
        score[..., : height // 2, :] = 0
        score[..., math.ceil(-height / 2) :, :] = 0

        score = torch.where(score <= -1e-7, score, score.min())
        score = torch.exp(score / (height * width))
        return score.squeeze(1)

    def _qatm(self, x: torch.Tensor, template_name) -> torch.Tensor:
        batch_size, img_row, img_col, template_row, template_col = x.size()
        x = x.view(batch_size, img_row * img_col, template_row * template_col)
        xm_img = x - torch.max(x, dim=1, keepdim=True).values
        xm_template = x - torch.max(x, dim=2, keepdim=True).values
        if self.positive_func == "softmax":
            img_conf = F.softmax(self.alpha_img * xm_img, dim=1)
            template_conf = F.softmax(self.alpha_template * xm_template, dim=2)
        elif self.positive_func == "exp":
            img_conf = torch.exp(self.alpha_img * xm_img)
            template_conf = torch.exp(self.alpha_template * xm_template)

            if self.max_sum:
                img_sum = img_conf.sum(dim=1, keepdim=True)
                temp_sum = template_conf.sum(dim=1, keepdim=True)

                img_sum_max = getattr(self, f"{template_name}_img_sum_max", None)
                if img_sum_max is None:
                    img_sum_max = img_sum.max(0, keepdim=True).values
                else:
                    img_sum_max = (
                        torch.cat((img_sum, img_sum_max)).max(0, keepdim=True).values
                    )
                img_sum_max = img_sum_max.detach()
                setattr(self, f"{template_name}_img_sum_max", img_sum_max)

                temp_sum_max = getattr(self, f"{template_name}_temp_sum_max", None)
                if temp_sum_max is None:
                    temp_sum_max = temp_sum.max(0, keepdim=True).values
                else:
                    temp_sum_max = (
                        torch.cat((temp_sum, temp_sum_max)).max(0, keepdim=True).values
                    )
                temp_sum_max = temp_sum_max.detach()
                setattr(self, f"{template_name}_temp_sum_max", temp_sum_max)

                img_conf = img_conf / img_sum_max
                template_conf = template_conf / temp_sum_max
        elif self.positive_func == "sigmoid":
            img_conf = torch.sigmoid(self.alpha_img * xm_img)
            template_conf = torch.sigmoid(self.alpha_template * xm_template)
        else:
            assert False
        confidence = torch.sqrt(img_conf * template_conf)
        _, ind3 = torch.topk(confidence, 1)
        ind1, ind2 = torch.meshgrid(
            torch.arange(batch_size, device=x.device),
            torch.arange(img_row * img_col, device=x.device),
        )
        values = confidence[ind1.flatten(), ind2.flatten(), ind3.flatten()]
        values = torch.reshape(values, [batch_size, img_row, img_col, 1])
        return values

    def match_single(
        self,
        img: np.ndarray,
        template_name: str,
        reset_img_features: bool = True,
        use_max_pool: bool = False,
        nms_iou_thresh: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param img: (height x width x channels) in RGB order, dtype of uint8
        :returns: (scores (height x width), center_xs, center_ys)
        """
        thresh = self.threshes[template_name]
        if self.tracker is not None:
            img = self.tracker.process_img(img, template_name)

        with torch.no_grad():
            img = self.feature_extractor.transform(img).unsqueeze(0)
            score = self(img.to(self.device), template_name, reset_img_features)

        if self.tracker is not None:
            score = self.tracker.process_score(score, template_name)

        if use_max_pool:
            pooled = torch.nn.functional.max_pool2d(
                score, kernel_size=3, stride=1, padding=1
            )
            _, center_ys, center_xs = (
                ((score == pooled) & (score >= thresh)).nonzero().transpose(0, 1)
            )
            center_xs, center_ys = map(
                lambda t: t.cpu().numpy(), (center_xs, center_ys)
            )
        else:
            detect_idx = score >= thresh
            box_scores = score[detect_idx]
            y, x = detect_idx.squeeze().nonzero().transpose(0, 1)
            height, width = self.templates[template_name].shape[:2]
            boxes = torch.stack((x, y, x + width, y + height), axis=-1)
            keep_idx = (
                torchvision.ops.nms(boxes.float(), box_scores, nms_iou_thresh)
                .cpu()
                .numpy()
            )
            center_xs = x[keep_idx]
            center_ys = y[keep_idx]

        return score.squeeze().cpu().numpy(), center_xs, center_ys

    def match_multi(
        self,
        img: torch.Tensor,
        template_name: str,
        reset_img_features: bool = True,
        use_max_pool: bool = False,
        nms_iou_thresh: float = 0.3,
        argmax: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param img: (n x channels x height x width) in RGB order, float32
        :param thresh: threshold at or above which a score is considerated a detection
          (after pooling or NMS)
        :returns: (scores (height x width), center_xs, center_ys)
        """
        thresh = self.threshes[template_name]

        # convert to NCHW
        # img = img.transpose(1, 3).transpose(2, 3).to(self.device, dtype=torch.float32)
        # img /= 255
        # img -= self._mean
        # img /= self._std

        score = self(img, template_name, reset_img_features)

        if self.tracker is not None:
            score = self.tracker.process_score(score, template_name)

        if argmax:
            maxes = score.max(-1).values.max(-1).values[..., None, None]
            max_idx = (score == maxes).nonzero().float()
            img_idx = max_idx[:, 0]
            last_max_idx = (img_idx[1:] > img_idx[:-1]).nonzero().squeeze(1)
            last_max_idx = torch.cat(
                (last_max_idx, torch.tensor([len(img_idx) - 1], device=self.device))
            )
            idx = max_idx[last_max_idx][:, 1:]
            # idx = idx[maxes.squeeze(-1).squeeze(-1) >= thresh]
            idx[maxes.squeeze(-1).squeeze(-1) < thresh] = float("nan")
        elif use_max_pool:
            assert False
        else:
            assert False

        # idx is like [[img_idx, y, x], ...]
        return score, idx

    def match(
        self, imgs: torch.Tensor, pool: bool = False, nms_iou_thresh: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        :param imgs: already transformed, if necessary
        :returns: (scores (n_images x n_templates x height x width), detect_locs)
        """
        # TODO - trackers (for multiple images, already pre-processed, as tensors)

        scores = []
        for i, template_name in enumerate(self.templates):
            score = self(imgs, template_name, reset_img_features=i == 0)
            if self.tracker is not None:
                score = self.tracker.process_score(score, template_name)
            scores.append(score)
        scores = torch.stack(scores, 1)

        if self._idx is None:
            height, width = scores.shape[2:]
            self._idx = torch.arange(height * width, device=self.device)
            self._idx = self._idx.reshape(1, 1, height, width)

        if pool:
            # this way was slightly faster than pooling and checking score == pooled
            _, max_idx = torch.nn.functional.max_pool2d(
                scores, 3, stride=1, padding=1, return_indices=True
            )
            detect_locs = (
                (self._idx == max_idx) & (scores >= self.stacked_threshes)
            ).nonzero()
        else:
            detect_locs = (scores >= self.stacked_threshes).nonzero()

        return scores, detect_locs

    @staticmethod
    def _norm(
        x1: torch.Tensor, x2: Optional[torch.Tensor] = None, min_std: float = 1e-10
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Normalize x1 and x2 using their joint mean and standard deviation.

        The normalization is in the channel dimension (mean / std across height and
        width for each channel of each image).

        :param x1: (batch x channels x height x width)
        :param x2: (batch x channels x height' x width'). Must match x1 in batch and
          channels.
        :param min_std: this is added to the standard deviation to prevent dividing by 0.
        """
        B, C, H, W = x1.shape
        x1 = x1.view(B, C, H * W)
        if x2 is None:
            concat = x1
        else:
            _, _, h, w = x2.shape
            x2 = x2.view(B, C, h * w)
            concat = torch.cat((x1, x2), dim=2)
        x_mean = torch.mean(concat, dim=2, keepdim=True)
        x_std = torch.std(concat, dim=2, keepdim=True) + min_std
        x1 = (x1 - x_mean) / x_std
        x1 = x1.view(B, C, H, W)
        if x2 is None:
            return x1
        x2 = (x2 - x_mean) / x_std
        x2 = x2.view(B, C, h, w)
        return x1, x2

    def do_bn(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        """
        B, C, H, W = x1.shape
        x1 = x1.view(B, C, 1, H * W)
        _, _, h, w = x2.shape
        x2 = x2.view(B, C, 1, h * w)
        concat = torch.cat((x1, x2), dim=-1)
        concat = self.bn_layer(concat)
        x1 = concat[..., : H * W]
        x2 = concat[..., H * W :]
        x1 = x1.view(B, C, H, W)
        x2 = x2.view(B, C, h, w)
        return x1, x2


def initialize_bilinear_interpolation(conv_layer: nn.Module) -> None:
    # from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/networks/resnet_dcn.py
    # copyright Microsoft, modifications copyright Xingyi Zhou and Nathan Hunt, all MIT licensed
    w = conv_layer.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class SymbolicMappingMulti:
    """
    Designed for use with AgentActionSafetyChecker.

    This uses detector.match_multi instead of match_single to process observations from
    all environments at once. Observations are also used directly instead of taking the
    environment as input and extracting the desired observation from it.
    """

    def __init__(
        self, detector, nms_iou_thresh: float = 0.3,
    ):
        self.detector = detector
        self.nms_iou_thresh = nms_iou_thresh

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """
        :param obs: shape (n x ...)
        :returns: shape (n x n_templates * n_features). The order for the second
          dimension is temp1_y, temp1_x, temp2_y, ... .
        """
        all_idx = []
        for template_idx, (template_name, (height, width)) in enumerate(
            self.detector.template_shapes.items()
        ):
            _, idx = self.detector.match_multi(
                obs,
                template_name,
                reset_img_features=template_idx == 0,
                nms_iou_thresh=self.nms_iou_thresh,
            )
            idx[:, 0] -= height // 2
            idx[:, 1] -= width // 2
            all_idx.append(idx)
        sym_feat = torch.cat(all_idx, dim=-1)
        return sym_feat
