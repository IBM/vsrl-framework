from typing import Optional

import numpy as np


def gen_separated_points(
    n_points: int,
    sep: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    initial_points: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Randomly generate `n_points` points from [lb, ub] that are at least `sep` apart.
    Note - this shouldn't be used to generate a large number of points as the runtime is
    quadratic in the number of points.

    :param lower_bounds: (d,)
    :param upper_bounds: (d,)
    :param initial_points: (k x d) use this if there are some points whose positions you
      already know so that the random points will be sufficiently far away from them.
      The returned points *include* `initial_points` (so the number of random points is
      `n_points - k` if `initial_points` is given).
    :returns: (n_points x d) Any `initial_points` will be included first, in the order
      they were given.
    """
    sep2 = sep ** 2

    if initial_points is None:
        points = np.empty((n_points, len(lower_bounds)))
        next_point_idx = 0
    else:
        if len(initial_points) == n_points:
            return initial_points
        points = np.empty((n_points - len(initial_points), len(lower_bounds)))
        points = np.concatenate((initial_points, points))
        next_point_idx = len(initial_points)

    while True:  # probably only need this loop once, but to be safe
        maybe_points = np.random.uniform(lower_bounds, upper_bounds, (n_points * 2, 2))
        if next_point_idx == 0:  # initial point has nothing to compare with
            points[0] = maybe_points[0]
            maybe_points = maybe_points[1:]
            next_point_idx = 1
            if next_point_idx == n_points:
                return points

        for point in maybe_points:
            if (((points[:next_point_idx] - point) ** 2).sum(-1) <= sep2).any():
                continue
            points[next_point_idx] = point
            next_point_idx += 1
            if next_point_idx == n_points:
                return points


class EpisodeTerminatedException(Exception):
    def __init__(self):
        msg = "This episode has terminated; call reset before stepping again."
        super().__init__(msg)
