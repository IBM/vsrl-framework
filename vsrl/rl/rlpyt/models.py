#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from typing import List, Tuple

import torch
import torchvision
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from torch import nn


def intersperse(interspersed_item, items) -> list:
    """Put `interspersed_item` between each of the elements of `items`."""
    if not items:
        return []
    ret = [items[0]]
    for item in items[1:]:
        ret.append(interspersed_item)
        ret.append(item)
    return ret


def make_layers_from_args(Module: nn.Module, args_lists) -> List[nn.Module]:
    """
    :param args_lists: a list of lists of arguments for `Module`. E.g. [in_size, out_size]
      where in_size and out_size have the same length
    :returns: layers such that `layers[i] = Module(*[args_list[i] for args_list in args_lists])`
    """
    return [Module(*args) for args in zip(*args_lists)]


class OracleModel(nn.Module):
    def __init__(
        self, n_inputs: int, action_dim: int, categorical: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.categorical = categorical
        shared_sizes = [100, 100, 100, 100, 100]
        action_sizes = [100] + [action_dim]
        value_sizes = [100] + [1]
        # activation = nn.ReLU(inplace=True)
        activation = nn.Tanh()

        in_sizes = [n_inputs] + shared_sizes[:-1]
        layers = make_layers_from_args(nn.Linear, [in_sizes, shared_sizes])
        self.feature_extractor = nn.Sequential(
            *intersperse(activation, layers), activation
        )

        in_sizes = [shared_sizes[-1]] + action_sizes[:-1]
        layers = make_layers_from_args(nn.Linear, [in_sizes, action_sizes])
        layers = intersperse(activation, layers)

        if categorical:
            self.action_predictor = nn.Sequential(*layers, nn.Softmax(dim=1))
        else:
            self.mu_predictor = nn.Sequential(*layers, Squash())
            self.log_std_predictor = nn.Linear(shared_sizes[-1], action_dim)

        in_sizes = [shared_sizes[-1]] + value_sizes[:-1]
        layers = make_layers_from_args(nn.Linear, [in_sizes, value_sizes])
        self.value_predictor = nn.Sequential(*intersperse(activation, layers))

    def forward(self, obs, extract_sym_features: bool = False):
        lead_dim, T, B, _ = infer_leading_dims(obs, 1)

        feats = self.feature_extractor(obs.view(T * B, -1))
        value = self.value_predictor(feats).squeeze(-1)  # squeezing seems expected?

        sym_feats = obs if extract_sym_features else None

        if self.categorical:
            action_dist = self.action_predictor(feats)
            action_dist, value = restore_leading_dims(
                (action_dist, value), lead_dim, T, B
            )
            return action_dist, value, sym_feats
        else:
            mu = self.mu_predictor(feats)
            log_std = self.log_std_predictor(feats)
            mu, log_std, value = restore_leading_dims(
                (mu, log_std, value), lead_dim, T, B
            )
            return mu, log_std, value, sym_feats


class VisionModel(nn.Module):
    def __init__(
        self,
        img_shape,
        n_vector_obs: int,
        action_dim: int,
        categorical: bool = True,
        sym_extractor=None,
    ):
        """
        :param action_dim: The number of actions if `categorical` otherwise the number
          of dimensions in the continuous action space.
        """
        super().__init__()
        h, w, c = img_shape
        self.action_dim = action_dim
        self.n_vector_obs = n_vector_obs
        self.categorical = categorical

        layers = (
            nn.Conv2d(c, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        test_img = torch.empty(c, h, w)[None]
        n_features = nn.Sequential(*layers)(test_img).numel()
        self.feature_extractor = nn.Sequential(*layers)
        self.linear = nn.Linear(n_features + n_vector_obs, 256)
        self.relu = nn.ReLU(inplace=True)

        if categorical:
            self.action_predictor = nn.Sequential(
                nn.Linear(256, action_dim), nn.Softmax(dim=-1),
            )
        else:
            self.mu_predictor = nn.Sequential(nn.Linear(256, action_dim), Squash())
            self.log_std_predictor = nn.Linear(256, action_dim)
        self.value_predictor = nn.Linear(256, 1)
        self.sym_extractor = sym_extractor

    def forward(self, obs, extract_sym_features: bool = False):
        imgs = obs.img
        lead_dim, T, B, img_shape = infer_leading_dims(imgs, 3)
        imgs = imgs.view(T * B, *img_shape)
        # NHWC -> NCHW
        imgs = imgs.transpose(3, 2).transpose(2, 1)

        # TODO: don't do uint8 -> float32 conversion twice (in detector and here)
        if self.sym_extractor is not None and extract_sym_features:
            sym_feats = self.sym_extractor(imgs[:, -1:])
        else:
            sym_feats = None

        # convert from [0, 255] uint8 to [0, 1] float32
        imgs = imgs.to(torch.float32)
        imgs.mul_(1.0 / 255)

        feats = self.feature_extractor(imgs)
        vector_obs = obs.vector.view(-1, obs.vector.shape[-1])
        feats = torch.cat((feats, vector_obs), -1)
        feats = self.relu(self.linear(feats))
        value = self.value_predictor(feats).squeeze(-1)  # squeezing seems expected?

        if self.categorical:
            action_dist = self.action_predictor(feats)
            action_dist, value = restore_leading_dims(
                (action_dist, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*action_dist.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return action_dist, value, sym_feats
        else:
            mu = self.mu_predictor(feats)
            log_std = self.log_std_predictor(feats)
            mu, log_std, value = restore_leading_dims(
                (mu, log_std, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*mu.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return mu, log_std, value, sym_feats


class VisionModel2(nn.Module):
    def __init__(
        self, img_shape, action_dim: int, categorical: bool = True, sym_extractor=None,
    ):
        """
        :param action_dim: The number of actions if `categorical` otherwise the number
          of dimensions in the continuous action space.
        """
        super().__init__()
        h, w, c = img_shape
        self.action_dim = action_dim
        self.categorical = categorical

        layers = (
            nn.Conv2d(c, 16, kernel_size=8, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )

        resnet18 = torchvision.models.resnet18(pretrained=False)

        if c == 2:
            self.normalize_mean = torch.tensor([0.449]).reshape(1, -1, 1, 1)
            self.normalize_std = torch.tensor([0.226]).reshape(1, -1, 1, 1)
            # only difference is 1 input channel instead of 3
            self.conv1 = nn.Conv2d(
                2, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.normalize_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(
                1, -1, 1, 1
            )
            self.normalize_std = torch.tensor([0.229, 0.224, 0.225]).reshape(
                1, -1, 1, 1
            )
            assert False, f"replicate conv1's weights to support 6? ({c}) in channels"
            self.conv1 = resnet18.conv1

        self.bn1 = resnet18.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = nn.Conv2d(64, 64, kernel_size=5, stride=4, bias=False)
        self.layer3 = nn.Conv2d(64, 32, kernel_size=3, stride=2, bias=False)
        self.flatten = nn.Flatten()

        test_img = torch.empty(c, h, w)[None]
        n_features = self.layer3(
            self.layer2(self.layer1(self.maxpool(self.conv1(test_img))))
        ).numel()
        self.linear = nn.Linear(n_features, 256)

        if categorical:
            self.action_predictor = nn.Sequential(
                nn.Linear(256, action_dim), nn.Softmax(dim=-1),
            )
        else:
            self.mu_predictor = nn.Sequential(nn.Linear(256, action_dim), Squash())
            self.log_std_predictor = nn.Linear(256, action_dim)
        self.value_predictor = nn.Linear(256, 1)
        self.sym_extractor = sym_extractor

    def forward(self, imgs, extract_sym_features: bool = False):
        lead_dim, T, B, img_shape = infer_leading_dims(imgs, 3)
        imgs = imgs.view(T * B, *img_shape)
        # NHWC -> NCHW
        imgs = imgs.transpose(3, 2).transpose(2, 1)

        # TODO: don't do uint8 -> float32 conversion twice (in detector and here)
        if self.sym_extractor is not None and extract_sym_features:
            sym_feats = self.sym_extractor(imgs[:, -1:])
        else:
            sym_feats = None

        # convert from [0, 255] uint8 to [0, 1] float32
        imgs = imgs.to(torch.float32)
        imgs.mul_(1.0 / 255)
        m = self.normalize_mean.to(imgs.device)
        s = self.normalize_std.to(imgs.device)
        imgs.sub_(m).div_(s)

        feats = self.conv1(imgs)
        feats = self.bn1(feats)
        feats = self.relu(feats)
        feats = self.maxpool(feats)
        feats = self.layer1(feats)
        feats = self.relu(feats)
        feats = self.layer2(feats)
        feats = self.relu(feats)
        feats = self.layer3(feats)
        feats = self.relu(feats)
        feats = self.flatten(feats)
        feats = self.linear(feats)
        feats = self.relu(feats)

        value = self.value_predictor(feats).squeeze(-1)  # squeezing seems expected?

        if self.categorical:
            action_dist = self.action_predictor(feats)
            action_dist, value = restore_leading_dims(
                (action_dist, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*action_dist.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return action_dist, value, sym_feats
        else:
            mu = self.mu_predictor(feats)
            log_std = self.log_std_predictor(feats)
            mu, log_std, value = restore_leading_dims(
                (mu, log_std, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*mu.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return mu, log_std, value, sym_feats


class _ImpalaBlock(nn.Module):
    def __init__(self, n_channels_in: int, n_channels_out: int):
        super().__init__()
        self.n_channels_in = n_channels_in
        self.n_channels_out = n_channels_out
        kernel_size = 3
        padding = 1
        self.conv1 = nn.Conv2d(
            n_channels_in, n_channels_out, kernel_size, padding=padding
        )
        self.pool = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.res1 = _ImpalaResBlock(n_channels_out)
        self.res2 = _ImpalaResBlock(n_channels_out)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x


class _ImpalaResBlock(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        kernel_size = 3
        padding = 1
        self.relu = nn.ReLU()
        self.relu_inplace = nn.ReLU()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding)

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.relu_inplace(x)
        x = self.conv2(x)
        x += inputs
        return x


class _ImpalaCNN(nn.Module):
    def __init__(
        self,
        img_shape: Tuple[int, int, int],
        n_extra_feats: int = 0,
        n_outputs: int = 256,
    ):
        super().__init__()
        self.n_outputs = n_outputs
        h, w, c = img_shape
        self.block1 = _ImpalaBlock(c, 16)
        self.block2 = _ImpalaBlock(16, 32)
        self.block3 = _ImpalaBlock(32, 32)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        test_img = torch.empty(c, h, w)[None]
        n_feats = self.block3(self.block2(self.block1(test_img))).numel()
        self.linear = nn.Linear(n_feats + n_extra_feats, self.n_outputs)

    def forward(self, x, extra_obs=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(x)
        x = self.flatten(x)
        if extra_obs is not None:
            x = torch.cat((x, extra_obs), -1)
        x = self.linear(x)
        x = self.relu(x)
        return x


class ImpalaVisModel(nn.Module):
    def __init__(
        self,
        img_shape,
        action_dim: int,
        categorical: bool = True,
        sym_extractor=None,
        n_vector_obs: int = 0,
    ):
        """
        :param action_dim: The number of actions if `categorical` otherwise the number
          of dimensions in the continuous action space.
        """
        super().__init__()
        self.action_dim = action_dim
        self.categorical = categorical
        self.vector_obs_size = n_vector_obs

        self.feature_extractor = _ImpalaCNN(img_shape, self.vector_obs_size)
        n_feats = 256 + self.vector_obs_size

        if categorical:
            self.action_predictor = nn.Sequential(
                nn.Linear(n_feats, action_dim), nn.Softmax(dim=-1),
            )
        else:
            self.mu_predictor = nn.Sequential(nn.Linear(n_feats, action_dim), Squash())
            self.log_std_predictor = nn.Linear(n_feats, action_dim)
        self.value_predictor = nn.Linear(n_feats, 1)
        self.sym_extractor = sym_extractor

    def forward(self, obs, extract_sym_features: bool = False):
        imgs = obs.img
        lead_dim, T, B, img_shape = infer_leading_dims(imgs, 3)
        imgs = imgs.view(T * B, *img_shape)
        # NHWC -> NCHW
        imgs = imgs.transpose(3, 2).transpose(2, 1)

        # TODO: don't do uint8 -> float32 conversion twice (in detector and here)
        if self.sym_extractor is not None and extract_sym_features:
            sym_feats = self.sym_extractor(imgs[:, -1:])
        else:
            sym_feats = None

        # convert from [0, 255] uint8 to [0, 1] float32
        imgs = imgs.to(torch.float32)
        imgs.mul_(1.0 / 255)

        vector_obs = obs.vector.view(-1, obs.vector.shape[-1])
        feats = self.feature_extractor(imgs, vector_obs)
        feats = torch.cat((feats, vector_obs), -1)
        value = self.value_predictor(feats).squeeze(-1)  # squeezing seems expected?

        if self.categorical:
            action_dist = self.action_predictor(feats)
            action_dist, value = restore_leading_dims(
                (action_dist, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*action_dist.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return action_dist, value, sym_feats
        else:
            mu = self.mu_predictor(feats)
            log_std = self.log_std_predictor(feats)
            mu, log_std, value = restore_leading_dims(
                (mu, log_std, value), lead_dim, T, B
            )
            if self.sym_extractor is not None and extract_sym_features:
                # have to "restore dims" for sym_feats manually...
                sym_feats = sym_feats.reshape((*mu.shape[:-1], -1))
                # sym_feats = torch.rand_like(action_dist)
            return mu, log_std, value, sym_feats


class ImpalaSacModel(nn.Module):
    def __init__(
        self,
        observation_shape,
        action_size: int,
        categorical: bool = False,
        sym_extractor=None,
    ):
        """
        :param action_dim: The number of actions if `categorical` otherwise the number
          of dimensions in the continuous action space.
        """
        if categorical:
            raise ValueError("This model does not support categorical action spaces.")

        super().__init__()
        self.action_dim = action_size

        self.feature_extractor = _ImpalaCNN(
            observation_shape.img, observation_shape.vector[0]
        )
        n_feats = self.feature_extractor.n_outputs

        mlp_activation = nn.ReLU()
        mlp_hidden_size = 128
        self.pi = nn.Sequential(
            nn.Linear(n_feats, mlp_hidden_size),
            mlp_activation,
            # first half are mu, second half are log_std
            nn.Linear(mlp_hidden_size, 2 * action_size),
        )
        self.q1 = nn.Sequential(
            nn.Linear(n_feats + action_size, mlp_hidden_size),
            mlp_activation,
            nn.Linear(mlp_hidden_size, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(n_feats + action_size, mlp_hidden_size),
            mlp_activation,
            nn.Linear(mlp_hidden_size, 1),
        )
        self.sym_extractor = sym_extractor

    def forward(self, obs, q_or_pi: str, extract_sym_features: bool = False):
        """
        :returns: (q1, q2, sym_features) or (mu, log_std, sym_features)
        """
        if q_or_pi == "q":
            obs, actions = obs
        imgs = obs.img
        lead_dim, T, B, img_shape = infer_leading_dims(imgs, 3)
        imgs = imgs.view(T * B, *img_shape)
        # NHWC -> NCHW
        imgs = imgs.transpose(3, 2).transpose(2, 1)

        # TODO: don't do uint8 -> float32 conversion twice (in detector and here)
        if self.sym_extractor is not None and extract_sym_features:
            sym_feats = self.sym_extractor(imgs[:, -1:])
        else:
            sym_feats = None

        # convert from [0, 255] uint8 to [0, 1] float32
        imgs = imgs.to(torch.float32)
        imgs = imgs.mul(1.0 / 255)

        vector_obs = obs.vector.view(-1, obs.vector.shape[-1])
        feats = self.feature_extractor(imgs, vector_obs)

        if q_or_pi == "q":
            feats = torch.cat((feats, actions.view(-1, actions.shape[-1])), -1)
            r1 = self.q1(feats)
            r2 = self.q2(feats)
        elif q_or_pi == "pi":
            r = self.pi(feats)
            r1 = r[:, : self.action_dim]  # mu
            r2 = r[:, self.action_dim :]  # log_std
        else:
            raise ValueError("q_or_pi must be 'q' or 'pi'.")

        r1, r2 = restore_leading_dims((r1, r2), lead_dim, T, B)
        if self.sym_extractor is not None and extract_sym_features:
            # have to "restore dims" for sym_feats manually...
            sym_feats = sym_feats.reshape((*r1.shape[:-1], -1))
        return r1, r2, sym_feats


class Squash(nn.Module):
    def __call__(self, x):
        """Squash x so all entries are in [-1, 1]."""
        return -1 + 2 / (1 + torch.exp(-0.5 * x))
