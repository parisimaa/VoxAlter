
    import numpy as np
    import pyaudio
    import wave
    import struct
    import math

    from myfunctions import clip16

    wavefile = 'author.wav'

    # Open wave file (should be mono channel)
    wf = wave.open( wavefile, 'rb' )

    #----- user data -----
    n1 = 441
    n2 = n1
    s_win = 1024
    FS, DAFx_in = wavfile.read('author.wav')

    #----- initialize windows, arrays, etc -----
    w1 = np.hanning(s_win) # analysis window
    w2 = w1 # synthesis window
    L = len(DAFx_in)
    DAFx_in = np.hstack([np.zeros(s_win), DAFx_in, np.zeros(s_win - (L % n1))]) / np.max(np.abs(DAFx_in))
    DAFx_out = np.zeros(len(DAFx_in))
    #UUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUUU
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

    #----- listening and saving the output -----
    DAFx_out = DAFx_out[s_win : s_win + L] / np.max(np.abs(DAFx_out))
    wavfile.write('author_robot.wav', FS, DAFx_out.astype(np.float32))
