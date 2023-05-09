import wave
import pyaudio
import struct
import math
import numpy as np
import tkinter as Tk
import librosa
from scipy import signal

# Clip function for values exceeding the 2^(WIDTH-1) limit
def clip_fn(x):
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768

    return x

############################### Audio ###############################

# Importing audio file
wavfile = 'author.wav'
print("Playing wav:", wavfile)

# Open wave file
wf = wave.open( wavfile, 'rb')

# Read wave file properties
RATE        = wf.getframerate()     # Frame rate (frames/second)
WIDTH       = wf.getsampwidth()     # Number of bytes per sample
LEN         = wf.getnframes()       # Signal length
CHANNELS    = wf.getnchannels()     # Number of channels

print('The file has %d channel(s).'         % CHANNELS)
print('The file has %d frames/second.'      % RATE)
print('The file has %d frames.'             % LEN)
print('The file has %d bytes per sample.'   % WIDTH)

# Open audio device:
p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)

stream = p.open(
    format    = PA_FORMAT,
    channels  = CHANNELS,
    rate      = RATE,
    input     = False,
    output    = True)

# Output file to be stored
output_wav = 'q1.wav'
wf_output = wave.open( output_wav, 'w')

# Setting the same parameters as the input file
wf_output.setframerate(RATE)
wf_output.setsampwidth(WIDTH)
wf_output.setnchannels(CHANNELS)

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

pitch_val = Tk.Scale(root, label = 'pitch', variable = pitch, 
                  from_ = -10, to = 10, resolution=1)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)

B_quit.pack(side = Tk.BOTTOM, fill = Tk.X)
pitch_val.pack(side = Tk.LEFT)

############################### Echo ###############################

# delay = 5 # milliseconds
HOP_SIZE = 256
BLOCKSIZE = 2048
n_fft = 512
bins_per_octave = 12
n_steps = 0

def pitch_fn(level):
    # pitch shift parameters
    if level > 0:
        high = True
        n_steps = level
    elif level < 0:
        high = False
        n_steps = abs(level)
    else:
        return 2.0 ** (-abs(level) / bins_per_octave)

    # if pitch higher than normal
    if high:
        s_rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    # if pitch lower than normal
    else:
        s_rate = 0.5 ** (-float(n_steps) / bins_per_octave)

    return s_rate

# Separate buffers for recursive and non-recursive mode
output_block = BLOCKSIZE * [0]

def norm(x, nbits):
    # nbits is the width
    x = np.array(x, dtype=np.float32)
    x /= 2 ** (nbits - 1)
    return x

def denorm(x, nbits):
    # nbits is the width
    x *= 2 ** (nbits - 1)
    return x

def time_stretch(y, s_rate, n_fft):
    
    y = np.array(y)
    
    # STFT
    stft = librosa.stft(norm(y, WIDTH))

    # Stretch by phase vocoding
    stretched_stft = librosa.phase_vocoder(
        stft,
        rate=s_rate,
        hop_length=n_fft//4
    )

    # Predict the length of y_stretch
    len_stretch = int(round(y.shape[-1] / s_rate))

    # Invert the STFT
    y_stretch = librosa.istft(stretched_stft, dtype=float, 
                           length=len_stretch)

    return y_stretch

def resample(y, orig_sr, target_sr):

    # Getting the last index
    axis = -1

    # Ratio defines how much we are stretching the sample
    ratio = float(target_sr) / orig_sr

    # 
    n_samples = int(np.ceil(y.shape[axis] * ratio))

    y_hat = signal.resample(y, n_samples, axis=axis)

    # Match dtypes
    return np.asarray(y_hat, dtype=y.dtype)

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


# input_bytes = wf.readframes(BLOCKSIZE)
input_bytes = wf.readframes(BLOCKSIZE)

while len(input_bytes) == BLOCKSIZE * WIDTH and CONTINUE:
    root.update()

    # Unpacking input block
    x0 = struct.unpack('h' * BLOCKSIZE, input_bytes)

    s_rate = pitch_fn(pitch.get())

    # Time Stretch
    output_block = time_stretch(x0, s_rate, n_fft)

    # Resampling
    output_block = resample(output_block, orig_sr = float(RATE)/s_rate,
                            target_sr = RATE)

    # Fix Length
    output_block = denorm(fix_length(output_block, len(x0)), WIDTH)
    output_block = [int(i) for i in output_block]
    output_bytes = struct.pack('h'*BLOCKSIZE, *output_block)
    

    stream.write(output_bytes)
    wf_output.writeframes(output_bytes)

    # input_bytes = wf.readframes(BLOCKSIZE)
    input_bytes = wf.readframes(BLOCKSIZE)

    # If the audio ends, it will rewind the audio to the start
    if len(input_bytes) < BLOCKSIZE * WIDTH:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKSIZE)


stream.stop_stream()
stream.close()
p.terminate()
wf.close()
wf_output.close()