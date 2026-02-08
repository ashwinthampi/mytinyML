#maxpool2d layer implementation
#implements 2d max pooling operation for convolutional neural networks
#forward pass: takes maximum value in each pool_size x pool_size region (fully vectorized)
#backward pass: routes gradient only to the argmax location (fully vectorized)
#reduces spatial dimensions, improves translation invariance, and speeds up computation
#supports GPU acceleration via CuPy backend

import numpy as np
import backend

class MaxPool2D:
    def __init__(self, pool_size: int = 2, stride: int | None = None):
        #pool size (assumed square: pool_size x pool_size)
        self.pool_size = pool_size
        #stride for pooling (defaults to pool_size for non-overlapping pooling)
        self.stride = stride if stride is not None else pool_size

        #cache for backward pass (stores which position had the max value)
        self._X = None
        self._argmax = None  #stores flat index of max position for vectorized backward

    #forward pass: take max in each pool_size x pool_size region (fully vectorized, no spatial loops)
    def forward(self, X):
        xp = backend.xp

        #X shape: (N, C, H, W)
        #output shape: (N, C, H_out, W_out)

        #shape assertion
        assert X.ndim == 4, f"Input must be 4D (N, C, H, W), got {X.ndim}D"

        #cache input for backward pass
        self._X = X

        N, C, H_in, W_in = X.shape
        p = self.pool_size
        s = self.stride

        #calculate output dimensions
        H_out = (H_in - p) // s + 1
        W_out = (W_in - p) // s + 1

        #validate that dimensions divide cleanly (helps catch accidental configs)
        assert (H_in - p) % s == 0, \
            f"Height dimension doesn't divide cleanly: (H={H_in} - pool_size={p}) % stride={s} != 0"
        assert (W_in - p) % s == 0, \
            f"Width dimension doesn't divide cleanly: (W={W_in} - pool_size={p}) % stride={s} != 0"

        if s == p:
            #non-overlapping case: pure reshape, no loops needed
            X_reshaped = X.reshape(N, C, H_out, p, W_out, p)
            X_blocks = X_reshaped.transpose(0, 1, 2, 4, 3, 5).reshape(N, C, H_out, W_out, p * p)

            #max and argmax over the last axis (pool window)
            self._argmax = xp.argmax(X_blocks, axis=4).astype(np.int32)
            Y = xp.max(X_blocks, axis=4)
        else:
            #overlapping case: use stride_tricks to create a view of all pool windows
            strides_X = X.strides
            view_shape = (N, C, H_out, W_out, p, p)
            view_strides = (
                strides_X[0], strides_X[1],
                s * strides_X[2], s * strides_X[3],
                strides_X[2], strides_X[3]
            )
            X_windows = xp.lib.stride_tricks.as_strided(X, shape=view_shape, strides=view_strides)
            X_blocks = X_windows.reshape(N, C, H_out, W_out, p * p)

            self._argmax = xp.argmax(X_blocks, axis=4).astype(np.int32)
            Y = xp.max(X_blocks, axis=4)

        return Y

    #backward pass: route gradient only to the argmax location (fully vectorized, no spatial loops)
    def backward(self, dY):
        xp = backend.xp

        #assertion to ensure forward was called first
        assert self._X is not None, "forward() must be called before backward()"
        assert self._argmax is not None, "forward() must be called before backward()"

        #validate dY shape
        assert dY.ndim == 4, f"dY must be 4D (N, C, H, W), got {dY.ndim}D"
        assert dY.shape == self._argmax.shape, f"dY shape {dY.shape} doesn't match cached output shape {self._argmax.shape}"

        X = self._X
        N, C, H_in, W_in = X.shape
        _, _, H_out, W_out = dY.shape
        p = self.pool_size
        s = self.stride

        #initialize gradient (zeros everywhere, will set values at argmax positions)
        dX = xp.zeros_like(X)

        #convert argmax flat indices to (h_offset, w_offset) within each pool window
        max_h_offset = self._argmax // p  #(N, C, H_out, W_out)
        max_w_offset = self._argmax % p   #(N, C, H_out, W_out)

        #build grids for all output positions at once
        h_out_grid = xp.arange(H_out)[None, None, :, None]  #(1, 1, H_out, 1)
        w_out_grid = xp.arange(W_out)[None, None, None, :]  #(1, 1, 1, W_out)

        #calculate absolute input positions for every output position
        h_in_positions = h_out_grid * s + max_h_offset  #(N, C, H_out, W_out)
        w_in_positions = w_out_grid * s + max_w_offset  #(N, C, H_out, W_out)

        #build batch and channel index arrays
        n_idx = xp.arange(N)[:, None, None, None]  #(N, 1, 1, 1)
        c_idx = xp.arange(C)[None, :, None, None]  #(1, C, 1, 1)

        #broadcast all to (N, C, H_out, W_out)
        n_idx = xp.broadcast_to(n_idx, (N, C, H_out, W_out))
        c_idx = xp.broadcast_to(c_idx, (N, C, H_out, W_out))

        #scatter-add gradients to argmax positions (handles potential overlaps for overlapping pooling)
        xp.add.at(dX, (n_idx, c_idx, h_in_positions, w_in_positions), dY)

        return dX

    #return the layer parameters (empty for maxpool since it has no learnable parameters)
    def parameters(self) -> dict:
        return {}
