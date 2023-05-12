from scipy.signal.windows import hann
import numpy as np
from scipy.linalg import norm


def pad_center(data, size, axis = -1):
    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    return np.pad(data, lengths, mode='constant')

def overlap_add(y, ytmp, hop_length):
    n_fft = ytmp.shape[-2]
    N = n_fft
    for frame in range(ytmp.shape[-1]):
        sample = frame * hop_length
        if N > y.shape[-1] - sample:
            N = y.shape[-1] - sample
        
        # print(y[..., sample : (sample + N)], ytmp[..., :N, frame])
        y[..., sample : (sample + N)] += ytmp[..., :N, frame][0]

    return y


def window_sumsquare(window, n_frames, hop_length, win_length, n_fft, dtype):

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = hann(win_length)
    win_sq = pad_center(win_sq ** 2, size=n_fft)

    # Fill the envelope
    n = len(x)
    n_fft = len(win_sq)
    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]

    return x

def fix_length(data, size):
    axis = -1
    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths)

    return data

def tiny(x):
    # Make sure we have an array view
    x = np.asarray(x)

    # Only floating types generate a tiny
    if np.issubdtype(x.dtype, np.floating) or np.issubdtype(
        x.dtype, np.complexfloating
    ):
        dtype = x.dtype
    else:
        dtype = np.dtype(np.float32)

    return np.finfo(dtype).tiny



def istft(stft_matrix, length):
    
    n_fft = 1024
    win_length = 512
    hop_length = win_length // 4
    center = True
    pad_mode = "constant"
    max_mem = 2**8 * 2**10

    ifft_window = hann(win_length)

    # Pad the window out to n_fft size
    ifft_window = pad_center(ifft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    axes = -2
    ndim = 1 + stft_matrix.ndim
    axes_tup = tuple([axes])

    shape = [1] * ndim
    for i, axi in enumerate(axes_tup):
        shape[axi] = ifft_window.shape[i]
    
    ifft_window = ifft_window.reshape(shape)

    # For efficiency, trim STFT frames according to signal length if available
    padded_length = length + 2 * (n_fft // 2)
    n_frames = min(stft_matrix.shape[-1], int(np.ceil(padded_length / hop_length)))

    dtype = np.complex128

    shape = list(stft_matrix.shape[:-2])
    expected_signal_len = n_fft + hop_length * (n_frames - 1)

    expected_signal_len -= 2 * (n_fft // 2)

    shape.append(expected_signal_len)

    y = np.zeros(shape, dtype=dtype)

    # First frame that does not depend on padding
    #  k * hop_length - n_fft//2 >= 0
    # k * hop_length >= n_fft // 2
    # k >= (n_fft//2 / hop_length)

    start_frame = int(np.ceil((n_fft // 2) / hop_length))

    # Do overlap-add on the head block
    ytmp = ifft_window * np.fft.irfft(stft_matrix[..., :start_frame], n=n_fft, axis=-2)

    shape[-1] = n_fft + hop_length * (start_frame - 1)
    head_buffer = np.zeros(shape, dtype=dtype)

    head_buffer = overlap_add(head_buffer, ytmp, hop_length)

    # If y is smaller than the head buffer, take everything
    if y.shape[-1] < shape[-1] - n_fft // 2:
        y[..., :] = head_buffer[..., n_fft // 2 : y.shape[-1] + n_fft // 2]
    else:
        # Trim off the first n_fft//2 samples from the head and copy into target buffer
        y[..., : shape[-1] - n_fft // 2] = head_buffer[..., n_fft // 2 :]

    # This offset compensates for any differences between frame alignment
    # and padding truncation
    offset = start_frame * hop_length - n_fft // 2

    n_columns = int(max_mem // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize))
    n_columns = max(n_columns, 1)

    frame = 0
    for bl_s in range(start_frame, n_frames, n_columns):
        bl_t = min(bl_s + n_columns, n_frames)

        # invert the block and apply the window function
        ytmp = ifft_window * np.fft.irfft(stft_matrix[..., bl_s:bl_t], n=n_fft, axis=-2)

        # Overlap-add the istft block starting at the i'th frame
        y = overlap_add(y[..., frame * hop_length + offset :], ytmp, hop_length)

        frame += bl_t - bl_s

    # Normalize by sum of squared window
    ifft_window_sum = window_sumsquare(
        window=ifft_window,
        n_frames=n_frames,
        win_length=win_length,
        n_fft=n_fft,
        hop_length=hop_length,
        dtype=dtype,
    )

    if center:
        start = n_fft // 2
    else:
        start = 0

    ifft_window_sum = fix_length(ifft_window_sum[..., start:], size=y.shape[-1])

    approx_nonzero_indices = ifft_window_sum > tiny(ifft_window_sum)

    y[..., approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return y