
import numpy as np
import wave
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import pickle as pk
import random

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split


def process(filename):
    sample_rate, audio_signal = wavfile.read(filename)  # File assumed to be in the same directory
    print(audio_signal.shape)
    audio_signal= np.mean(audio_signal[:50000], axis= 1)
    signal_length = len(audio_signal)
    N = audio_signal.shape[0]
    L = N / sample_rate
    print(f'Audio length: {L:.2f} seconds')
    frame_size = 0.005
    frame_stride = 0.001  # overlap
    # convert from seconds to samples
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    # make sure we have atleast one frame
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    # pad to make sure that all frames ahve equal number of samples without truncating any samples from the orignal signals
    pad_signal = np.append(audio_signal, z)


    freqs, times, Sx = signal.spectrogram(audio_signal, fs=sample_rate, window='hanning',
                                          nperseg=num_frames, noverlap=num_frames - 100,
                                          detrend=False, scaling='spectrum')
    f, ax = plt.subplots(figsize=(4.8, 2.4))
    ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
    ax.set_ylabel('Frequency [kHz]')
    ax.set_xlabel('Time [s]')
    #plt.show()

    return audio_signal

def classify(audio_signal):
    x = [[random.randint(0, 5) for x in range(2)] for x in range(10)]
    y = [0 for x in range(10)]

    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

    #training the model
    model = Sequential()
    model.add(Dense(audio_signal.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(audio_signal.shape[1], activation='softmax'))
    #
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])


    file= open("model.pk", "wb")
    pk.dump(model, file)
    file.close()

    score = model.evaluate(x_test, y_test, batch_size=32, verbose= 0)
    return score



audio_signal= process("watermeloncracktest.wav")
classify(audio_signal)

#mfcc= process("watermeloncracktest.wav")
#view_mfcc_spectrogram(mfcc)


print(audio_signal.shape)
