#data augmentation utilities for image classification
#provides random shift, rotation, and a convenience pipeline function
#all functions operate on batches of images in (N, C, H, W) format
#pure numpy implementation (no scipy/PIL dependencies)
#optimized: vectorized across batch dimension (no per-image Python loops)

import numpy as np

def random_shift(X: np.ndarray, max_shift: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly shift images by up to max_shift pixels in each direction.

    X: (N, C, H, W) batch of images
    Returns: (N, C, H, W) shifted images (zero-padded)

    Optimized: groups images by shift value (max 25 groups for max_shift=2)
    instead of looping per image (N iterations).
    """
    if rng is None:
        rng = np.random.default_rng()

    N, C, H, W = X.shape
    X_aug = np.zeros_like(X)

    #generate all random shifts at once (vectorized)
    dx_all = rng.integers(-max_shift, max_shift + 1, size=N)
    dy_all = rng.integers(-max_shift, max_shift + 1, size=N)

    #group images by shift value and process each group in one vectorized op
    #max (2*max_shift+1)^2 = 25 groups for max_shift=2, vs N=256 per-image loops
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            mask = (dx_all == dx) & (dy_all == dy)
            if not np.any(mask):
                continue

            #source slices (region of original image to copy)
            src_y = slice(max(0, -dy), min(H, H - dy))
            src_x = slice(max(0, -dx), min(W, W - dx))
            #destination slices (where to paste in output)
            dst_y = slice(max(0, dy), min(H, H + dy))
            dst_x = slice(max(0, dx), min(W, W + dx))

            #apply shift to all images with this (dx, dy) at once
            X_aug[mask, :, dst_y, dst_x] = X[mask, :, src_y, src_x]

    return X_aug


def random_rotation(X: np.ndarray, max_angle: float = 15.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly rotate images by up to max_angle degrees using bilinear interpolation.

    X: (N, C, H, W) batch of images
    Returns: (N, C, H, W) rotated images (pure numpy, no scipy/PIL)

    Optimized: computes all N rotations simultaneously using broadcasting
    instead of looping per image.
    """
    if rng is None:
        rng = np.random.default_rng()

    N, C, H, W = X.shape

    center_y, center_x = H / 2.0, W / 2.0

    #precompute coordinate grids (shared across all images): (H, W) each
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    #generate all angles at once: (N,)
    angles = rng.uniform(-max_angle, max_angle, size=N).astype(np.float32) * (np.pi / 180.0)
    cos_a = np.cos(angles)  #(N,)
    sin_a = np.sin(angles)  #(N,)

    #center-relative coordinates: (H, W)
    dx_grid = xx - center_x
    dy_grid = yy - center_y

    #inverse rotation: map output coords to input coords for all images at once
    #broadcasting: (N, 1, 1) * (H, W) -> (N, H, W)
    cos_a_3d = cos_a[:, None, None]
    sin_a_3d = sin_a[:, None, None]

    src_x = cos_a_3d * dx_grid + sin_a_3d * dy_grid + center_x  #(N, H, W)
    src_y = -sin_a_3d * dx_grid + cos_a_3d * dy_grid + center_y  #(N, H, W)

    #bilinear interpolation indices: (N, H, W)
    x0 = np.floor(src_x).astype(np.intp)
    y0 = np.floor(src_y).astype(np.intp)
    x1 = x0 + 1
    y1 = y0 + 1

    #fractional parts for weighting: (N, H, W)
    wa = (x1 - src_x) * (y1 - src_y)
    wb = (src_x - x0) * (y1 - src_y)
    wc = (x1 - src_x) * (src_y - y0)
    wd = (src_x - x0) * (src_y - y0)

    #clip coordinates to valid range: (N, H, W)
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    #mask for out-of-bounds pixels: (N, H, W)
    mask = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)

    #batch index for fancy indexing: (N, 1, 1) broadcasts to (N, H, W)
    n_idx = np.arange(N)[:, None, None]

    #interpolate all images at once per channel (C=1 for MNIST, so just 1 iteration)
    X_aug = np.zeros_like(X)
    for c in range(C):
        img_c = X[:, c]  #(N, H, W)
        result = (wa * img_c[n_idx, y0c, x0c] +
                  wb * img_c[n_idx, y0c, x1c] +
                  wc * img_c[n_idx, y1c, x0c] +
                  wd * img_c[n_idx, y1c, x1c])
        X_aug[:, c] = result * mask

    return X_aug


def augment_batch(X: np.ndarray, rng: np.random.Generator | None = None,
                  shift: bool = True, rotate: bool = True,
                  max_shift: int = 2, max_angle: float = 15.0) -> np.ndarray:
    """Apply a pipeline of augmentations to a batch.

    Applies enabled augmentations in sequence: shift then rotate.
    Returns augmented copy (does not modify input).
    """
    if rng is None:
        rng = np.random.default_rng()

    X_aug = X.copy()
    if shift:
        X_aug = random_shift(X_aug, max_shift=max_shift, rng=rng)
    if rotate:
        X_aug = random_rotation(X_aug, max_angle=max_angle, rng=rng)

    return X_aug
