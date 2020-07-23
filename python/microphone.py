import time

import numpy as np
import pyaudio


def start_stream(configs, callback):
    fps = configs['fps']
    mic_rate = configs['mic_rate']

    p = pyaudio.PyAudio()
    frames_per_buffer = int(mic_rate / fps)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=mic_rate,
                    input=True,
                    frames_per_buffer=frames_per_buffer)
    overflows = 0
    prev_ovf_time = time.time()
    while True:
        try:
            y = np.fromstring(stream.read(frames_per_buffer, exception_on_overflow=False), dtype=np.int16)
            y = y.astype(np.float32)
            stream.read(stream.get_read_available(), exception_on_overflow=False)
            callback(y)
        except IOError:
            overflows += 1
            if time.time() > prev_ovf_time + 1:
                prev_ovf_time = time.time()
                print('Audio buffer has overflowed {} times'.format(overflows))
    stream.stop_stream()
    stream.close()
    p.terminate()
