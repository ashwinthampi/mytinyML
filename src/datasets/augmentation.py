#data augmentation utilities for image classification
#provides random shift, rotation, and a convenience pipeline function
#all functions operate on batches of images in (N, C, H, W) format
#pure numpy implementation (no scipy/PIL dependencies)

import numpy as np

def random_shift(X: np.ndarray, max_shift: int = 2, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly shift images by up to max_shift pixels in each direction.

    X: (N, C, H, W) batch of images
    Returns: (N, C, H, W) shifted images (zero-padded)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, C, H, W = X.shape
    X_aug = np.zeros_like(X)

    for i in range(N):
        dx = rng.integers(-max_shift, max_shift + 1)
        dy = rng.integers(-max_shift, max_shift + 1)

        #source slices (region of original image to copy)
        src_y = slice(max(0, -dy), min(H, H - dy))
        src_x = slice(max(0, -dx), min(W, W - dx))
        #destination slices (where to paste in output)
        dst_y = slice(max(0, dy), min(H, H + dy))
        dst_x = slice(max(0, dx), min(W, W + dx))

        X_aug[i, :, dst_y, dst_x] = X[i, :, src_y, src_x]

    return X_aug


def random_rotation(X: np.ndarray, max_angle: float = 15.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Randomly rotate images by up to max_angle degrees using bilinear interpolation.

    X: (N, C, H, W) batch of images
    Returns: (N, C, H, W) rotated images (pure numpy, no scipy/PIL)
    """
    if rng is None:
        rng = np.random.default_rng()

    N, C, H, W = X.shape
    X_aug = np.zeros_like(X)

    center_y, center_x = H / 2.0, W / 2.0

    #precompute coordinate grids (shared across all images)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    for i in range(N):
        angle = rng.uniform(-max_angle, max_angle) * np.pi / 180.0
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        #inverse rotation: map output coords back to input coords
        src_x = cos_a * (xx - center_x) + sin_a * (yy - center_y) + center_x
        src_y = -sin_a * (xx - center_x) + cos_a * (yy - center_y) + center_y

        #bilinear interpolation
        x0 = np.floor(src_x).astype(int)
        y0 = np.floor(src_y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        #fractional parts for weighting
        wa = (x1 - src_x) * (y1 - src_y)
        wb = (src_x - x0) * (y1 - src_y)
        wc = (x1 - src_x) * (src_y - y0)
        wd = (src_x - x0) * (src_y - y0)

        #clip coordinates to valid range
        x0c = np.clip(x0, 0, W - 1)
        x1c = np.clip(x1, 0, W - 1)
        y0c = np.clip(y0, 0, H - 1)
        y1c = np.clip(y1, 0, H - 1)

        #mask for out-of-bounds pixels
        mask = (src_x >= 0) & (src_x < W) & (src_y >= 0) & (src_y < H)

        for c in range(C):
            img = X[i, c]
            result = wa * img[y0c, x0c] + wb * img[y0c, x1c] + wc * img[y1c, x0c] + wd * img[y1c, x1c]
            X_aug[i, c] = result * mask

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
