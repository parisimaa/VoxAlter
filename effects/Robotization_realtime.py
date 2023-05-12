import wave
import pyaudio
import struct
import math
import numpy as np
import tkinter as Tk
from scipy import signal
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

RATE        = 44100     # Frame rate (frames/second)
WIDTH       = 2         # Number of bytes per sample
CHANNELS    = 1         # Number of channels
DURATION    = 10

R = 512
Nfft = 512

print('The file has %d channel(s).'         % CHANNELS)
print('The file has %d frames/second.'      % RATE)
#print('The file has %d frames.'             % LEN)
print('The file has %d bytes per sample.'   % WIDTH)

# Open audio device:
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)

stream = p.open(
    format    = pyaudio.paFloat32,
    channels  = CHANNELS,
    rate      = RATE,
    input     = True,
    output    = True,
    frames_per_buffer= R)

# Output file to be stored
output_wav = 'robo.wav'
wf_output = wave.open( output_wav, 'w')

# Setting the same parameters as the input file
wf_output.setframerate(RATE)
wf_output.setsampwidth(WIDTH)
wf_output.setnchannels(CHANNELS)

############################### BLOCK ############################## 
#BLOCKLEN = 1024
# output_block = R* [0]
# Number of blocks in wave file
#num_blocks = int(math.floor(LEN/BLOCKLEN))
# num_blocks = int(RATE / R * DURATION)
# input_bytes = wf.readframes(BLOCKSIZE)
#input_bytes = wf.readframes(BLOCKLEN)

############################### Robo #############################
# Define the STFT and inverse STFT functions
def stft(x, R, Nfft):
    hop_size = R//2
    w = np.hanning(R)
    stft_matrix = np.empty((Nfft//2 + 1, 1 + (len(x) - R) // hop_size), dtype=np.complex64)
    for i in range(stft_matrix.shape[1]):
        segment = x[i * hop_size : i * hop_size + R]
        windowed_segment = segment * w
        stft_matrix[:, i] = np.fft.rfft(windowed_segment, Nfft)

    return stft_matrix

def inv_stft(X, R, N):
    hop_size = R//2
    w = np.hanning(R)
    x = np.zeros(N, dtype=np.float32)
    for i in range(X.shape[1]):
        segment = np.fft.irfft(X[:, i], R)
        x[i * hop_size : i * hop_size + R] += segment * w
    return x

def denorm(x, nbits):
    # nbits is the width
    x = 2 * (nbits*8 - 1)
    return x

############################## Main Loop ##########################
MAXVALUE = 2**15-1
# output_block = R * [0]
# for i in range(0, num_blocks):
is_running = True

# Define a function to quit the program
def quit_program():
    global is_running
    is_running = False


while is_running:

    # Read a chunk of audio from the microphone
    input_data = stream.read(R, exception_on_overflow=False)
    # Convert the input data from bytes to a numpy array of floats
    x = np.frombuffer(input_data, dtype=np.float32)
    # Apply a threshold to the input signal
    if np.max(np.abs(x)) < 0.01:
        x = np.zeros_like(x)  # Set the input to zero if it's too faint
    # Compute the STFT of the input signal
    X = stft(x, R, Nfft)
    # Set phase to zero in STFT-domain
    X2 = X * np.exp(1j * 2 * np.pi * np.random.rand(*X.shape))
    y2 = inv_stft(X2,R, len(x))
    if np.max(np.abs(y2)) > 0:
        y2 = y2 / np.max(np.abs(y2))
    y2 *= 0.2  # Apply the scaling factor from the slider
    # Convert the output signal to bytes
    output_bytes = y2.astype(np.float32).tobytes()

    stream.write(output_bytes)
    wf_output.writeframes(output_bytes)


stream.stop_stream()
stream.close()
p.terminate()
wf_output.close()