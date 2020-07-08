#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Classes and functions to visualize results of trained agents.

This is only intended to be used within a Jupyter notebook.
"""

import gzip
import time
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import holoviews as hv
import numpy as np
from IPython import display


class Visualizer:
    """A class to display a video stream (e.g. an RL agent's observations)."""

    def __init__(
        self,
        obs_transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        opts: Optional[Dict[str, Any]] = None,
        save_obs: bool = False,
        multi: bool = False,
        n_cols: int = 3,
        show: bool = True,
    ):
        if obs_transform is None:
            obs_transform = lambda x: x
        self.opts = opts or {}
        self.multi = multi
        self.n_cols = n_cols
        self.show = show
        self.obs = []
        self.save_obs = save_obs
        self.obs_transform = obs_transform
        self.update = None

    def _get_dmap_updater(self, obs: np.ndarray):
        example = {"img": obs[None]}  # add dimension so length=1 means 1 image
        stream = hv.streams.Buffer(example, length=1)
        Element = hv.RGB if obs.ndim - int(self.multi) == 3 else hv.Image
        if self.multi:

            def callback(obs):
                plot = None
                for ob in obs:
                    if plot is None:
                        plot = Element(ob).opts(**self.opts)
                    else:
                        plot += Element(ob).opts(**self.opts)
                if self.n_cols:
                    plot.cols(self.n_cols)
                return plot

        else:
            callback = lambda obs: Element(obs).opts(**self.opts)

        img_dmap = hv.DynamicMap(
            lambda data: callback(data["img"][0]), streams=[stream]
        )

        def update(data):
            stream.send({"img": data[None]})

        display.display(img_dmap)
        return update

    def __call__(self, obs: np.ndarray) -> None:
        obs = self.obs_transform(obs)
        if self.save_obs:
            self.obs.append(obs)

        if not self.show:
            return

        if self.update:
            self.update(obs)
        else:
            self.update = self._get_dmap_updater(obs)

    def save_video(self, fname: str, fps: int = 30, fourcc: str = "MP4V") -> None:
        """
        :param fname: should probably be a .mp4 file if using MP4V on Mac
        """
        import cv2

        cc = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(fname, cc, fps, self.obs[0].shape[:-1][::-1])
        for frame in self.obs:
            writer.write(frame[..., ::-1])  # convert to BGR
        writer.release()

    def save_frames(self, fname: str) -> None:
        """
        Save the collected observations to a .npy.gz file.

        See `save_frames` for more information.
        """
        save_frames(self.obs, fname)

    def replay(
        self,
        start_index: int = 0,
        end_index: int = -1,
        delay: float = 0,
        callback: Optional[Callable[[int], Any]] = None,
    ) -> None:
        """
        Replay the saved observations in a new video.

        :param start_index: index of the first saved observation to show
        :param end_index: index of the last saved observation to show. Negative values
          are allowed and are interpreted like normal for indexing in Python.
        :param delay: how many seconds to wait between showing each frame
        :param callback: function to call after each frame update; the only input given
          is an int specifying how many times the function has already been called
        """
        if not self.obs:
            raise ValueError(
                "Can't replay an empty list of observations. "
                "Make sure to set save_obs=True at init."
            )
        self.update = self._get_dmap_updater(self.obs[start_index])
        for i, obs in enumerate(self.obs[start_index + 1 : end_index]):
            if delay:
                time.sleep(delay)
            if callback:
                callback(i)
            self.update(obs)
        if callback:
            callback(i + 1)

    @classmethod
    def from_files(
        cls,
        fnames: Union[str, Sequence[str]],
        opts: Optional[Dict[str, Any]] = None,
        multi: bool = False,
        delay: float = 0,
    ):
        """
        If some arrays are longer (more frames) than others, all arrays will be padded to
        the length of the longest array. The "padding" frames will be the same as the
        final frame of the array, so shorter videos will appear to stop on their final
        frame. All videos are shown simultaneously.

        :param fnames: ".npy.gz" will be added at the end of each name if no extension
          is given; ".npy" is also allowed if given explicitly.
        :param opts: Holoviews options for the RGB or Image elements used for the videos.
        :param multi: if True, there should only be one input file but which contains
          frames for multiple videos. Otherwise, there can be either one file with one
          video saved or multiple files with one video each.
        :param delay: how many seconds to wait between showing each frame
        """
        if isinstance(fnames, str):
            fnames = [fnames]

        all_obs = []
        for fname in fnames:
            if not (fname.endswith(".npy") or fname.endswith(".npy.gz")):
                fname = fname + ".npy.gz"
            if fname.endswith(".gz"):
                with gzip.GzipFile(fname, "r") as f:
                    all_obs.append(np.load(f))
            else:
                all_obs.append(np.load(fname))
        max_len = max(map(len, all_obs))
        for i, obs in enumerate(all_obs):
            len_diff = max_len - len(obs)
            if len_diff:
                all_obs[i] = np.concatenate((obs, [obs[-1]] * len_diff))
        all_obs = np.stack(all_obs, axis=1).squeeze()
        visualizer = cls(opts=opts, multi=multi or len(fnames) > 1)
        for frame in all_obs:
            visualizer(frame)
            if delay:
                time.sleep(delay)


def visualize(
    agent, env, max_steps: int = 10_000, opts: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    :param save_states: whether to return a list of the environment state at each
      step. This can be used to load the environment back into the state it was in at
      that step (e.g. for debugging safety from a dangerous state).
    :param restore_state: state to which to restore the environment before having the
      agent interact with it. There may be errors if you try to restore to a state saved
      in a different process...
    :returns: a dict whose keys are a subset of {}
    """
    opts = opts or {}
    obs = env.reset()

    visualizer = Visualizer(opts=opts)
    visualizer(obs)

    for _ in range(max_steps):
        action = agent.act(obs)
        obs, _, done, _ = env.step(action)
        visualizer(obs)
        if done:
            break


def save_frames(frames: Union[np.ndarray, Sequence[np.ndarray]], fname):
    if not isinstance(frames, np.ndarray):
        frames = np.stack(frames)
    if not fname.endswith(".npy.gz"):
        fname += ".npy.gz"
    with gzip.GzipFile(fname, "w") as f:
        np.save(f, frames)


def updating_curve(
    x_axis_name: str = "step",
    y_axis_name: str = "mean",
    max_n_points: int = 1000,
    curve_opts: Optional[Dict[str, Any]] = None,
    area_opts: Optional[Dict[str, Any]] = None,
    plot_error: bool = True,
) -> Tuple[Any, Callable[[Sequence[float]], None]]:
    """
    Get a plot which shows a mean curve and shaded area (mean +/- std) region.
    The plot is initially empty, but a function is returned which, when called on an
    array or similar, will update the plot with the mean/std results from that data.
    Usage:
    ```python
    plot, update = updating_curve()
    plot
    # new cell
    # generate 100 points, each from the mean of 10 random numbers
    for i in range(100):
        update(np.random.randint(10))
    ```
    :param plot_error: if True, a curve with a shaded error region is plotted. If False,
      only the curve is plotted. If no error is required, the updating function should be
      called with just a single point (only the first given point will be plotted).
    :returns: plot, updating function
    """
    curve_opts = curve_opts or {}
    area_opts = {"alpha": 0.5, **(area_opts or {})}
    example = {x_axis_name: np.array([]), y_axis_name: np.array([])}
    if plot_error:
        example.update({"lb": np.array([]), "ub": np.array([])})
    stream = hv.streams.Buffer(example, length=max_n_points)
    curve_dmap = hv.DynamicMap(
        partial(hv.Curve, kdims=x_axis_name, vdims=y_axis_name), streams=[stream]
    )
    plot = curve_dmap.opts(**curve_opts)
    if plot_error:
        area_dmap = hv.DynamicMap(
            partial(hv.Area, kdims=x_axis_name, vdims=["lb", "ub"]), streams=[stream]
        )
        area_dmap.opts(**area_opts)
        plot *= area_dmap

    i = 0

    def update(data):
        nonlocal i
        mean = np.mean(data)
        data = {x_axis_name: [i], y_axis_name: [mean]}
        if plot_error:
            std = np.std(data)
            data["lb"] = [mean - std]
            data["ub"] = [mean + std]
        stream.send(data)
        i += 1

    return plot, update
