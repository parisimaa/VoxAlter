import wave
import pyaudio
import struct
import math
import numpy as np
import tkinter as Tk

# Clip function for values exceeding the 2^(WIDTH-1) limit
def clip_fn(x):
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768

    return x

############################### Audio ###############################

# Importing audio file
wavfile = 'audio_sample_01.wav'
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
# Variable to identify whether to use Recursive or non-recursive mode
REC_MODE = True

# Function to quit the program
def fun_quit():
    global CONTINUE
    print('Good bye')
    CONTINUE = False

# Function to switch between Recursive and non-recursive mode and display it on a label in the window
def rec_fn():
    global REC_MODE
    if REC_MODE == True:
        rec.set("Non-Recursive Mode")
        REC_MODE = False
    else:
        REC_MODE = True
        rec.set("Recursive Mode")
    

# Define Tkinter root
root = Tk.Tk()

# Variables to control a, T and mode toggle
a = Tk.DoubleVar()
T = Tk.DoubleVar()
rec = Tk.StringVar()

a.set(0.8)
T.set(0.05)
rec.set("Recursive Mode")

a_val = Tk.Scale(root, label = 'a', variable = a, 
                  from_ = 0.1, to = 0.9, resolution=0.1)
T_val = Tk.Scale(root, label = 'T', variable = T, 
                  from_ = 0.0, to = 0.3, resolution=0.01)
B_quit = Tk.Button(root, text = 'Quit', command = fun_quit)
B_rec = Tk.Button(root, text = 'Recursive/Non Recursive Mode Toggle', command = rec_fn)

B_quit.pack(side = Tk.BOTTOM, fill = Tk.X)
a_val.pack(side = Tk.LEFT)
T_val.pack(side = Tk.RIGHT)
B_rec.pack(side = Tk.TOP)

L1 = Tk.Label(root, textvariable = a)
L2 = Tk.Label(root, textvariable = T)
L3 = Tk.Label(root, textvariable = rec)

L1.pack(side = Tk.LEFT)
L2.pack(side = Tk.RIGHT)
L3.pack(side = Tk.TOP)


############################### Echo ############################### 

BLOCKSIZE = 2048
b0 = 1.0            # direct-path gain
N = int(RATE * T.get())   # delay in samples
BUFFER_LEN = 8192   # Kept a large buffer as we have a time varying element

# Separate buffers for recursive and non-recursive mode
x_buffer = BUFFER_LEN * [0]
y_buffer = BUFFER_LEN * [0]
output_block = BLOCKSIZE * [0]

# Initialize phase
kr = 0  # read index  # (equivalent to BUFFER_LEN evaluated circularly)
kw = N  # write index

# input_bytes = wf.readframes(BLOCKSIZE)
input_bytes = wf.readframes(BLOCKSIZE)

# Grad variables to smoothen the transition between values of a and T
a_prev = a.get()
T_prev = T.get()

while len(input_bytes) == BLOCKSIZE * WIDTH:
    root.update()

    # Get current values of the grad variables
    a_curr = a.get()
    T_curr = T.get()

    # Removing audible artifacts due to slider movement
    a_grad = np.linspace(a_prev, a_curr, BLOCKSIZE)
    T_grad = np.linspace(T_prev, T_curr, BLOCKSIZE)

    # Unpacking input block
    x0 = struct.unpack('h' * BLOCKSIZE, input_bytes)

    for n in range(BLOCKSIZE):
        # Compute output value
        # If we are in the Recursive mode
        if REC_MODE:
            # This condition is when we dial the value of T to zero
            # In this case, the output is essentially the direct gain input + a gain multiplied version of the input
            # When T = 0, it doesn't read any past values, it needs the current input value. I have approximated any value below 0.01 as 0
            # This follows the logic that the resolution on my slider is 0.01, so it can't be any value between 0.01 and 0
            if T_grad[n] < 0.01:
                output_block[n] = b0 * x0[n] + a_grad[n] * x0[n]
            else:
                output_block[n] = b0 * x0[n] + a_grad[n] * y_buffer[kr-int(RATE * T_grad[n])]
        # If we are in the non-Recursive mode
        else:
            if T_grad[n] < 0.01:
                output_block[n] = b0 * x0[n] + a_grad[n] * x0[n]
            else:
                output_block[n] = b0 * x0[n] + a_grad[n] * x_buffer[kr-int(RATE * T_grad[n])]

        # Clipped Output
        output_block[n] = int(clip_fn(output_block[n]))

        # Update buffer (pure delay)
        x_buffer[kw] = x0[n]
        y_buffer[kw] = output_block[n]

        # Update grad variables
        a_prev = a_curr
        T_prev = T_curr

        # Increment read index
        kr = kr + 1
        if kr == BUFFER_LEN:
            # End of buffer. Circle back to front.
            kr = 0

        # Increment write index    
        kw = kw + 1
        if kw == BUFFER_LEN:
            # End of buffer. Circle back to front.
            kw = 0

    # output_bytes = struct.pack('h' * BLOCKSIZE, *y0)
    output_bytes = struct.pack('h'*BLOCKSIZE, *output_block)

    stream.write(output_bytes)
    wf_output.writeframes(output_bytes)

    # input_bytes = wf.readframes(BLOCKSIZE)
    input_bytes = wf.readframes(BLOCKSIZE)

    # Condtion to quit the program
    if CONTINUE == False:
        break

    # If the audio ends, it will rewind the audio to the start
    if len(input_bytes) < BLOCKSIZE * WIDTH:
        wf.rewind()
        input_bytes = wf.readframes(BLOCKSIZE)


stream.stop_stream()
stream.close()
p.terminate()
wf.close()
wf_output.close()