# VSRL Envs

## Observations

The default observations are named tuples with two fields

* `img`: a `h x w x (2 * c)` array where `h, w, c` are the height, width, and number of channels of the environment. `c` can be 3 or 1 depending on whether the observations are grayscale. There are `2 * c` channels in the observations because the two most recent frames are both included (the last `c` channels correspond to the most recent frame).
* `vector`: any additional observations are included as a 1D array; the size of this varies by environment

### Arguments

Each environment has the following arguments affecting the observations

* `img_scale`: a factor by which to downscale all of the images (e.g. `img_scale = 2` means the observations will be half of their original size). We set this to 4 in our experiments.
* `grayscale`: whether to use RGB or grayscale images for the observations. We use grayscale images in our experiments.
* `oracle_obs`: if true, the observations are the symbolic states of the environment instead of images. Each observation will be a 1D array whose length and values are environment-specific. This is mainly for debugging and is not used in our experiments.
