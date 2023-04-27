import wave
import pyaudio
import struct
import numpy as np
import tkinter as Tk

# open the specified input file specified above
wavfile = 'audio_sample_01.wav'
input_file = wave.open( wavfile, 'rb')

# get properties of the i/p wav file
RATE        = input_file.getframerate()    
WIDTH       = input_file.getsampwidth()     
LEN         = input_file.getnframes()      
CHANNELS    = input_file.getnchannels()    

# open the output file in write mode and set its properties
wf_output = wave.open( 'Question1_output.wav', 'w')
wf_output.setframerate(RATE)
wf_output.setsampwidth(WIDTH)
wf_output.setnchannels(CHANNELS)

# Clip the value of x between -32768 and 32767
def clip_fn(x):
    if x > 32767:
        return 32767
    elif x < -32768:
        return -32768
    return x


p = pyaudio.PyAudio()
PA_FORMAT = p.get_format_from_width(WIDTH)

# open the output audio stream
stream = p.open(
    format    = PA_FORMAT,
    channels  = CHANNELS,
    rate      = RATE,
    input     = False,
    output    = True)


progress = True     # indicates if the program should progress running
recursive_mode = True     # indicates if the program is in recursive mode or not

# function to toggle between recursive and non-recursive mode
def toggleMode():
    global recursive_mode
    if recursive_mode == True:
        recursive.set("non-recursive")
        recursive_mode = False
    else:
        recursive_mode = True
        recursive.set("recursive")

# function to quit
def endExecution():
    global progress
    progress = False


# Define Tkinter root and the variables for the a and T sliders and the recursive mode label
root = Tk.Tk()
root.title("** Welcome **")
root.geometry("500x550")

a = Tk.DoubleVar()
T = Tk.DoubleVar()
recursive = Tk.StringVar()

# set the initial values of a and T params, set the initial state to recursive
a.set(0.5)
T.set(0.07)
recursive.set("recursive")

# create labels for displaying the current values of a, T and current mode
label1 = Tk.Label(root, text="Current a value = ")
label1.grid(row=0, column=0, padx=10, pady=10, sticky=Tk.W)

label2 = Tk.Label(root, text="Current T value = ")
label2.grid(row=1, column=0, padx=10, pady=10, sticky=Tk.W)

label3 = Tk.Label(root, text="Current Mode = ")
label3.grid(row=2, column=0, padx=10, pady=10, sticky=Tk.W)

label4 = Tk.Label(root, textvariable=a)
label4.grid(row=0, column=1, padx=10, pady=10, sticky=Tk.W)

label5 = Tk.Label(root, textvariable=T)
label5.grid(row=1, column=1, padx=10, pady=10, sticky=Tk.W)

label6 = Tk.Label(root, textvariable=recursive)
label6.grid(row=2, column=1, padx=10, pady=10, sticky=Tk.W)

# slider widget for a and T
slider_a = Tk.Scale(root, label='a', variable=a, from_=0.1, to=0.9, resolution=0.1)
slider_a.grid(row=0, column=2, padx=10, pady=10)
slider_T = Tk.Scale(root, label='T', variable=T, from_=0.0, to=0.3, resolution=0.01)
slider_T.grid(row=1, column=2, padx=10, pady=10)

# create buttons for quitting and toggling recursive mode
B_quit = Tk.Button(root, text='Quit', command=endExecution)
B_quit.grid(row=3, column=0, padx=10, pady=10, columnspan=3, sticky=Tk.W+Tk.E)
B_rec = Tk.Button(root, text='Toggle modes', command=toggleMode)
B_rec.grid(row=4, column=0, padx=10, pady=10, columnspan=3, sticky=Tk.W+Tk.E)


b0 = 1.0                  # direct-path gain
N = int(RATE * T.get())   # delay in samples
kr = 0  # read index
kw = N  # write index
# initialize variables for audio processing
BLOCKSIZE = 2048
BUFFER_LEN = 8192

input_delay_line = BUFFER_LEN*[0]     # stores the most recent BUFFER_LEN samples of the input signal
output_delay_line = BUFFER_LEN*[0]    # stores the corresponding delayed samples of the output signal
output_samples = BLOCKSIZE*[0]          # stores the output values after processing each block of the input


# read input audio in blocks of blocksize
input_bytes = input_file.readframes(BLOCKSIZE)

a_prev = a.get()
T_prev = T.get()

while len(input_bytes) == BLOCKSIZE * WIDTH:
    root.update()

    # convert the input audio bytes into an array of samples
    x0 = struct.unpack('h' * BLOCKSIZE, input_bytes)

    # Linear interpolation
    a_curr = a.get()
    T_curr = T.get()
    a_interp = np.linspace(a_prev, a_curr, BLOCKSIZE)
    T_interp = np.linspace(T_prev, T_curr, BLOCKSIZE)

    # process each sample in the block
    for n in range(BLOCKSIZE):
        if recursive_mode:  
            # Recursive filter
            if T_interp[n] == 0.0:
                # If the delay time is 0, the output is just the direct-path gain times the input
                output_samples[n] = b0*x0[n] + a_interp[n]*x0[n]
            else:
                # Otherwise, the output is the direct-path gain times the input plus the delayed signal
                output_samples[n] = b0*x0[n] + a_interp[n]*output_delay_line[kr-int(RATE * T_interp[n])]
        else:    
            # Non-recursive filter
            if T_interp[n] == 0.0:
                # If the delay time is 0, the output is just the direct-path gain times the input
                output_samples[n] = b0*x0[n] + a_interp[n]*x0[n]
            else:
                # Otherwise, the output is the direct-path gain times the input plus the delayed signal
                output_samples[n] = b0*x0[n] + a_interp[n]*input_delay_line[kr-int(RATE * T_interp[n])]

        # clip the output sample
        output_samples[n] = int(clip_fn(output_samples[n]))

        # update the delay buffer
        input_delay_line[kw] = x0[n]
        output_delay_line[kw] = output_samples[n]

        # increment the read index
        kr = kr + 1
        if kr == BUFFER_LEN:
            kr = 0  # If the read index has reached the end of the buffer, wrap around to the beginning (circular buffer)
        # increment the write index        
        kw = kw + 1
        if kw == BUFFER_LEN:
            kw = 0  # If the write index has reached the end of the buffer, wrap around to the beginning (circular buffer)
    
    # convert the output sample array into an array of bytes
    output_bytes = struct.pack('h'*BLOCKSIZE, *output_samples)

    # write the output audio data to the output file and to the audio output stream
    stream.write(output_bytes)
    wf_output.writeframes(output_bytes)

    # read next block of input audio
    input_bytes = input_file.readframes(BLOCKSIZE)

    # If the user has clicked the "Quit" button, break out of the loop
    if progress == False:
        break

    # If there is not enough input audio data left to fill a full block, rewind to the beginning of the input file and read the remaining audio data again (loop playback)
    if len(input_bytes) < BLOCKSIZE*WIDTH:
        input_file.rewind()
        input_bytes = input_file.readframes(BLOCKSIZE)

    # Update previous values
    a_prev = a_curr
    T_prev = T_curr

stream.stop_stream()
stream.close()
p.terminate()
wf_output.close()
input_file.close()