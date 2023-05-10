import wave
import pyaudio
import struct
import math
import numpy as np
import tkinter as Tk
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
# DAFx_in = np.frombuffer(wf.readframes(-1), dtype=np.int16)
#print(np.shape(DAFx_in))

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
output_wav = 'robo.wav'
wf_output = wave.open( output_wav, 'w')

# Setting the same parameters as the input file
wf_output.setframerate(RATE)
wf_output.setsampwidth(WIDTH)
wf_output.setnchannels(CHANNELS)

############################### BLOCK ############################## 
BLOCKLEN = 2048
output_block = BLOCKLEN * [0]
# Number of blocks in wave file
num_blocks = int(math.floor(LEN/BLOCKLEN))
# input_bytes = wf.readframes(BLOCKSIZE)
input_bytes = wf.readframes(BLOCKLEN)

############################### Robo #############################
n1 = 441
n2 = n1
s_win = 1024
#----- initialize windows, arrays, etc -----
w1 = np.hanning(s_win) # analysis window
w2 = w1 # synthesis window
def Robo(DAFx_in):
    L = len(DAFx_in)
    # Make sure that input is within the grain size
    DAFx_in = np.hstack([np.zeros(s_win), DAFx_in, np.zeros(s_win - (L % n1))]) / np.max(np.abs(DAFx_in))
    # Zero padding
    DAFx_out = np.zeros(len(DAFx_in)) 
    # This normalization ensures that the signal does not clip after processing.
    pin = 0
    pout = 0
    pend = len(DAFx_in) - s_win
    while pin < pend:
        grain = DAFx_in[pin : pin + s_win] * w1
        f = np.fft.fft(grain)
        r = np.abs(f)
        grain = np.fft.fftshift(np.real(np.fft.ifft(r))) * w2
        DAFx_out[pout : pout + s_win] += grain
        pin += n1
        pout += n2
    DAFx_out=DAFx_out[s_win : s_win + L] / np.max(np.abs(DAFx_out))
    return DAFx_out

def denorm(x, nbits):
    # nbits is the width
    x *= 2 ** (nbits*8 - 1)
    return x

############################## Main Loop ##########################

while len(input_bytes) == BLOCKLEN * WIDTH:

    # Unpacking input block
    x0 = struct.unpack('h' * BLOCKLEN, input_bytes)

    # ROBO
    output_block = Robo(x0)
    output_block = [clip_fn(int(denorm(i, WIDTH))) for i in output_block]

    # Convert values to binary data
    output_bytes = struct.pack('h'*BLOCKLEN, *output_block)
    
    stream.write(output_bytes)
    wf_output.writeframes(output_bytes)

    # input_bytes = wf.readframes(BLOCKLEN)
    input_bytes = wf.readframes(BLOCKLEN)


stream.stop_stream()
stream.close()
p.terminate()
wf.close()
wf_output.close()
