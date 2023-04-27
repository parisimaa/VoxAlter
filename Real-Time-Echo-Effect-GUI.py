# Importing libraries
import tkinter as Tk
from tkinter import ttk
import pyaudio
import wave
import struct

def clip16( x ):    
    # Clipping for 16 bits
    if x > 32767:
        x = 32767
    elif x < -32768:
        x = -32768
    else:
        x = x        
    return (x)

# Importing audio file
wavfile = 'audio_sample_01.wav'
print('Play the wave file %s.' % wavfile)

# Open the wave file
wf = wave.open(wavfile, 'rb')

# Read wave file properties
RATE = wf.getframerate()  # Frame rate (frames/second)
WIDTH = wf.getsampwidth()  # Number of bytes per sample
LEN = wf.getnframes()  # Signal length
CHANNELS = wf.getnchannels()  # Number of channels

print('The file has %d channel(s).' % CHANNELS)
print('The file has %d frames/second.' % RATE)
print('The file has %d frames.' % LEN)
print('The file has %d bytes per sample.' % WIDTH)

# parameters
b0 = 1.0  # direct-path gain
kw = 0  # write index

# Define Tkinter root
root = Tk.Tk()
root.title("Echo Effect")

# --------------------------------------------
# Parameters that user will change on scaler
# Define Tk variables
a = Tk.DoubleVar()
T = Tk.DoubleVar()
recursive_mode = Tk.IntVar(value =1)

# quit button
def fun_quit():
    global CONTINUE
    print('Good bye')
    CONTINUE = False

# Buffer to store past signal values. Initialize to zero.
BUFFER_LEN = 6000  # Set buffer length. Must be more than N!
buffer = BUFFER_LEN * [0]  # list of zeros

# Initialize Tk variables
global kr
# feed-forward gain (a)
a.set(0.8)
# Delay in seconds
T.set(0.05)
delay_sec = T.get()  # Get value of T
N = int(RATE * float(delay_sec))
# Initialize buffer indices
kr = BUFFER_LEN - N  # read index

# Define widgets
S_freq = Tk.Scale(root, label='Gain', variable=a, from_=0.0, to=1.0, tickinterval=0.1, resolution=0.1, length=300)
S_delay = Tk.Scale(root, label='Delay (s)', variable=T, from_=0.0, to=0.3, tickinterval=0.05, resolution=0.01, length=300)
R_nonrecursive = Tk.Radiobutton(root, text="Non-Recursive", variable=recursive_mode, value=1)
R_recursive = Tk.Radiobutton(root, text="Recursive", variable=recursive_mode, value=2)
B_quit = Tk.Button(root, text='Quit', command=fun_quit)

# Place widgets
S_freq.pack(side=Tk.TOP, pady=10)
S_delay.pack(side=Tk.TOP, pady=10)
R_nonrecursive.pack(side=Tk.LEFT, padx=10)
R_recursive.pack(side=Tk.LEFT, padx=10)
B_quit.pack(side=Tk.BOTTOM, pady=20)

# Open an output audio stream
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=False,
    output=True
)

# -------------------------------------------
CONTINUE = True
# Get first frame
input_bytes = wf.readframes(1)

print('* Start')
y0=0
# Instead of len(input_bytes) > 0 put CONTINUE to see if audio loop 
while len(input_bytes) > 0:
	root.update()
	# Convert string to number
	x0, = struct.unpack('h', input_bytes)
	# non-recursive
	if recursive_mode.get() == 1:
	    # y(n) = b0 x(n) + G x(n-N)
	    y0 = b0 * x0 + a.get() * buffer[kr]
	    buffer[kw] = x0
	elif recursive_mode.get() == 2:
	    # y(n) = b0 x(n) + G y(n-N)
	    y0 = b0 * x0 + a.get() * buffer[kr]
	    # Update buffer (pure delay)
	    buffer[kw] = y0
	    # Increment read index
	kr = kr + 1
	if kr == BUFFER_LEN:
	    # End of buffer. Circle back to front.
	    kr = 0
	kw = kw + 1
	if kw == BUFFER_LEN:
	# End of buffer. Circle back to front.
		kw = 0
	# Clip and convert output value to binary data
	output_bytes = struct.pack('h', int(clip16(y0)))

	# Write output to audio stream
	stream.write(output_bytes)

	if CONTINUE == False:
		break


	# Get next frame
	input_bytes = wf.readframes(1) 
print('* Finished')

stream.stop_stream()
stream.close()
p.terminate()
wf.close()
