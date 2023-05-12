import pyaudio
import struct
import math
import numpy as np
import tkinter as Tk
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from stft import stft_fn
from istft import istft

# Clip function for values exceeding the 2^(WIDTH-1) limit
def clip_fn(x):
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768

    return x

############################### Audio ###############################

# Parameters for audio streaming
CHANNELS = 1
RATE = 16000
WIDTH = 2

# Open audio device:
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)

stream = p.open(
    format    = PA_FORMAT,
    channels  = CHANNELS,
    rate      = RATE,
    input     = True,
    output    = True)


############################### GUI ############################### 

# Variable to stop execution of the main loop
CONTINUE = True

# Function to quit the program
def fun_quit():
    global CONTINUE
    print('Good bye')
    CONTINUE = False
    

# Define Tkinter root
root = Tk.Tk()

# Variables to control a, T and mode toggle
pitch = Tk.IntVar()
pitch.set(0)

# Slider for Pitch Level and Button to go back to main GUI
pitch_val = Tk.Scale(root, variable = pitch, 
                  from_ = -5, to = 5, resolution=1, orient=Tk.HORIZONTAL)
B_quit = Tk.Button(root, text = 'Back', command = fun_quit)

# Packing the button in the Tk Window
B_quit.pack(side = Tk.BOTTOM, fill = Tk.X)
pitch_val.pack(side = Tk.BOTTOM, fill = Tk.X)

# Label for pitch slider
pitch_label = Tk.Label(root, text="Pitch Level",font=("Ariel",10,"bold", "underline"))
pitch_label.pack(pady=20, side= Tk.BOTTOM, anchor="center")

############################### Pitch Shift ###############################

# Since we are block processing, block length to be set
BLOCKSIZE = 2048
# Number of FFTs to be generated with sample of size BLOCKSIZE
n_fft = 1024
# Conventional 12 bins per octave. 
bins_per_octave = 12
# Pitch will depend on number of steps. Initiliazing with 0.
n_steps = 0

# Function for the GUI to report changes in pitch level
def pitch_fn(level):
    """
    Divided into higher than normal and lower than normal with different equations for each category
    """
    # If level is higher than normal
    if level > 0: 
        n_steps = level
        return 2.0 ** (-float(n_steps) / bins_per_octave)
    # If level is lower than normal
    elif level < 0:
        n_steps = abs(level)
        return 0.5 ** (-float(n_steps) / bins_per_octave)
    # If it's 0
    else:
        return 2.0 ** (-abs(level) / bins_per_octave)

# Output Block
output_block = BLOCKSIZE * [0]

# Higher precision for better processing
def norm(x, nbits):
    # nbits is the width of the audio
    x = np.array(x, dtype=np.float32)
    x /= 2 ** (nbits*8 - 1)
    return x

# Denormalizing for the audio stream
def denorm(x, nbits):
    # nbits is the width of the audio
    x *= 2 ** (nbits*8 - 1)
    return x

# Phase vocoder function for pitch shift
def phase_vocoder(D, rate, hop_length):
    # Calculating Number of FFTs from input array
    n_fft = 2 * (D.shape[-2] - 1)

    # For iterating over the audio matrix
    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    # Create an empty output array
    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros_like(D, shape=shape)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

    # Phase accumulator; initialize to the first sample
    phase_acc = np.angle(D[..., 0])

    # Pad 0 columns to simplify boundary logic
    # This is not ideal and can be improved
    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for t, step in enumerate(time_steps):
        columns = D[..., int(step) : int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = np.mod(step, 1.0)
        mag = (1.0 - alpha) * np.abs(columns[..., 0]) + alpha * np.abs(columns[..., 1])

        # Store to output array
        z = np.cos(phase_acc) + 1j * np.sin(phase_acc)
        z *= mag
        d_stretch[..., t] = z
        
        # Compute phase advance
        dphase = np.angle(columns[..., 1]) - np.angle(columns[..., 0]) - phi_advance

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * np.pi * np.round(dphase / (2.0 * np.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch



def time_stretch(y, s_rate, n_fft):
    # Converting list to np.array
    y = np.array(y)
    
    # Generate STFT
    stft = stft_fn(norm(y, WIDTH))

    # Stretch by phase vocoding 
    stretched_stft = phase_vocoder(
        stft,
        rate=s_rate,
        hop_length=n_fft//4
    )

    # Predict the length of y_stretch
    len_stretch = int(round(y.shape[-1] / s_rate))

    # Invert the STFT
    y_stretch = istft(stretched_stft, len_stretch)

    return y_stretch


def resample(y, orig_sr, target_sr):

    # Getting the last index
    axis = -1

    # Ratio defines how much we are stretching the sample
    ratio = float(target_sr) / orig_sr

    # According to input pitch level, calculate number of samples after time stretch
    n_samples = int(np.ceil(y.shape[axis] * ratio))

    # Resample signal according to the new sample size using sicpy.signal
    y_hat = signal.resample(y, n_samples, axis=axis)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)

# Since resampling changes the size of the matrix
def fix_length(data, size):
    # Choosing the end axis
    axis = -1
    n = data.shape[axis]

    # If resultant matrix is bigger than input
    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    # If resultant matrix is smaller than input
    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return np.pad(data, lengths)

    # If the sizes match
    return data

############################### Plot ###############################

# Figure Initializations
my_fig = plt.figure()
my_plot = my_fig.add_subplot(1, 1, 1) 
# Canvas for the Window
canvas = FigureCanvasTkAgg(my_fig, master = root)
my_fig.canvas.draw()

# Add widgets
W1 = canvas.get_tk_widget()
W1.pack()

# Plotting the input signal
input_plot, = my_plot.plot([], [], color = 'blue', label='input')  # Create empty line
                        # x-data of plot (frequency)

# Plotting the output signal
output_plot, = my_plot.plot([],[], color='red', label='output')

# Plot parameters
# Limits for X and Y axes
plt.xlim(0, BLOCKSIZE)
plt.ylim(-10000, 10000)
# Label for X axis
plt.xlabel("Time (n)")

# X axis will always be the size of block we are passing
f = np.arange(0, BLOCKSIZE)
input_plot.set_xdata(f) 
output_plot.set_xdata(f)

# Set initial block of zeros as Y data
input_plot.set_ydata([0]*BLOCKSIZE) 
output_plot.set_ydata([0]*BLOCKSIZE)
plt.legend()

# Get frames from audio input stream
input_bytes = stream.read(BLOCKSIZE, exception_on_overflow = False)

while len(input_bytes) == BLOCKSIZE * WIDTH and CONTINUE:
    root.update()

    # Unpacking input block
    x0 = struct.unpack('h' * BLOCKSIZE, input_bytes)

    # Getting the level of pitch from the GUI
    s_rate = pitch_fn(pitch.get())

    # Time Stretch
    output_block = time_stretch(x0, s_rate, n_fft)

    # Resampling
    output_block = resample(output_block, orig_sr = float(RATE)/s_rate,
                            target_sr = RATE)

    # Fix Length
    output_block = denorm(fix_length(output_block, len(x0)), WIDTH)

    # Changing quantization and clipping
    output_block = [int(clip_fn(i)) for i in output_block]

    # Packing data to be sent through audio stream as output
    output_bytes = struct.pack('h'*BLOCKSIZE, *output_block)

    # Update input and output plot according to input data
    input_plot.set_ydata(x0)
    output_plot.set_ydata(output_block)
    my_fig.canvas.draw()

    stream.write(output_bytes)

    # Get frames from audio input stream
    input_bytes = stream.read(BLOCKSIZE, exception_on_overflow = False)   # BLOCKLEN = number of frames read


stream.stop_stream()
stream.close()
p.terminate()
plt.close()