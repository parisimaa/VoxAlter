import numpy as np
from scipy.signal.windows import hann
from numpy.lib.stride_tricks import as_strided

def pad_center(data, size, axis = -1):
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, mode='constant')

def frame(x, frame_length, hop_length):

    axis = -1
    writeable = False
    subok = False

    x = np.array(x, copy=False)

    # put our new within-frame axis at the end for now
    out_strides = x.strides + tuple([x.strides[axis]])

    # Reduce the shape on the framing axis
    x_shape_trimmed = list(x.shape)
    x_shape_trimmed[axis] -= frame_length - 1

    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    xw = as_strided(x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable)

    target_axis = axis - 1
    xw = np.moveaxis(xw, -1, target_axis)

    # Downsample along the target axis
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)

    return xw[tuple(slices)]

def stft_fn(y):
    n_fft = 1024
    win_length = 512
    hop_length = win_length // 4
    center = True
    pad_mode = "constant"
    max_mem = 2**8 * 2**10


    fft_window = hann(win_length)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    axes = -2
    ndim=1 + y.ndim
    axes_tup = tuple([axes])

    shape = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = fft_window.shape[i]
    
    fft_window = fft_window.reshape(shape)


    # Set up the padding array to be empty, and we'll fix the target dimension later
    padding = [(0, 0) for _ in range(y.ndim)]

    # How many frames depend on left padding?
    start_k = int(np.ceil(n_fft // 2 / hop_length))

    # What's the first frame that depends on extra right-padding?
    tail_k = (y.shape[-1] + n_fft // 2 - n_fft) // hop_length + 1

    if tail_k <= start_k:
        # If tail and head overlap, then just copy-pad the signal and carry on
        start = 0
        extra = 0
        padding[-1] = (n_fft // 2, n_fft // 2)
        y = np.pad(y, padding, mode=pad_mode)
    else:
        # If tail and head do not overlap, then we can implement padding on each part separately
        # and avoid a full copy-pad

        # "Middle" of the signal starts here, and does not depend on head padding
        start = start_k * hop_length - n_fft // 2
        padding[-1] = (n_fft // 2, 0)

        # +1 here is to ensure enough samples to fill the window
        # fixes bug #1567
        y_pre = np.pad(
            y[..., : (start_k - 1) * hop_length - n_fft // 2 + n_fft + 1],
            padding,
            mode=pad_mode,
        )
        y_frames_pre = frame(y_pre, frame_length=n_fft, hop_length=hop_length)
        # Trim this down to the exact number of frames we should have
        y_frames_pre = y_frames_pre[..., :start_k]

        # How many extra frames do we have from the head?
        extra = y_frames_pre.shape[-1]

        # Determine if we have any frames that will fit inside the tail pad
        if tail_k * hop_length - n_fft // 2 + n_fft <= y.shape[-1] + n_fft // 2:
            padding[-1] = (0, n_fft // 2)
            y_post = np.pad(
                y[..., (tail_k) * hop_length - n_fft // 2 :], padding, mode=pad_mode
            )
            y_frames_post = frame(
                y_post, frame_length=n_fft, hop_length=hop_length
            )
            # How many extra frames do we have from the tail?
            extra += y_frames_post.shape[-1]
        else:
            # In this event, the first frame that touches tail padding would run off
            # the end of the padded array
            # We'll circumvent this by allocating an empty frame buffer for the tail
            # this keeps the subsequent logic simple
            post_shape = list(y_frames_pre.shape)
            post_shape[-1] = 0
            y_frames_post = np.empty_like(y_frames_pre, shape=post_shape)


    dtype = np.complex128

    # Window the time series.
    y_frames = frame(y[..., start:], frame_length=n_fft, hop_length=hop_length)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)

    # This is our frequency dimension
    shape[-2] = 1 + n_fft // 2

    # If there's padding, there will be extra head and tail frames
    shape[-1] += extra


    stft_matrix = np.zeros(shape, dtype=dtype, order="F")


    # Fill in the warm-up
    if center and extra > 0:
        off_start = y_frames_pre.shape[-1]
        stft_matrix[..., :off_start] = np.fft.rfft(fft_window * y_frames_pre, axis=-2)

        off_end = y_frames_post.shape[-1]
        if off_end > 0:
            stft_matrix[..., -off_end:] = np.fft.rfft(fft_window * y_frames_post, axis=-2)
    else:
        off_start = 0

    n_columns = int(max_mem // (np.prod(y_frames.shape[:-1]) * y_frames.itemsize))
    n_columns = max(n_columns, 1)

    for bl_s in range(0, y_frames.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, y_frames.shape[-1])

        stft_matrix[..., bl_s + off_start : bl_t + off_start] = np.fft.rfft(
            fft_window * y_frames[..., bl_s:bl_t], axis=-2
        )
    return stft_matrix