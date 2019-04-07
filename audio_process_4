#32000
import numpy as np
import wave
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import dct
import matplotlib.pyplot as plt
import pickle as pk
import random
import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

#signal_length= 100000
#signal_length=50000

def process(filename, signal_length= 100000, dual_source= False):
    audio_signals= [] #list of np arrays, outer length 100, inner length 0.5 seconds
    sample_rate, audio_signal = wavfile.read(filename)  # File assumed to be in the same directory
    #print("signal", audio_signal.shape)
    if dual_source:
         audio_signal= np.mean(audio_signal, axis= 1)
    #print(audio_signal.shape)

    # #else:
    # #    audio_signal= audio_signal[:signal_length]
    N = audio_signal.shape[0]
    L = N / sample_rate
    #print("N: ", N)
    #print(f'Audio length: {L:.2f} seconds')
    #CHANGE
    for i in range(10,109):#(10,109):
        start= int((i-.12)*sample_rate)
        end= int(start+ sample_rate*.2)
        #CHANGE
        clip= audio_signal[start:end:20]
        # n= clip.shape[0]
        # L= n/ sample_rate
        # print(f'Audio length: {L:.2f} seconds')

        #print(clip.shape)
        #plt.figure(1)
        #plt.title('Signal Wave...')
        #plt.plot(clip)
        #plt.show()
        frame_size = 0.005
        frame_stride = 0.001  # overlap
        # convert from seconds to samples
        frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
        # make sure we have atleast one frame
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        # pad to make sure that all frames ahve equal number of samples without truncating any samples from the orignal signals
        pad_signal = np.append(clip, z)
        audio_signals.append(clip)
    print(audio_signals)


    #freqs, times, Sx = signal.spectrogram(audio_signal, fs=sample_rate, window='hanning',
    #                                      nperseg=num_frames, noverlap=num_frames - 100,
    #                                      detrend=False, scaling='spectrum')
    #f, ax = plt.subplots(figsize=(4.8, 2.4))
    #ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
    #ax.set_ylabel('Frequency [kHz]')
    #ax.set_xlabel('Time [s]')
    #plt.show()


    return audio_signals


def classify(x, y, num_samples, signal_length=100000):
    #x = [1,2,3,4,5,6,7,8,9,10]# [x for x in range(100)]# [[random.randint(0, 5) for x in range(2)] for x in range(100)]
    #x = np.array([[1 for i in range(audio_signal.shape[0])]for j in range(num_samples)])
    #x = np.array(x_set) do this earlier
    print("x input shape:", x.shape)
    #y = np.array([[1] for x in range(num_samples)])
    print("y ********************")
    print(y)
    print("y shape", y.shape)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=.2)

    model = Sequential()
    model.add(Dense(25, activation='sigmoid', input_shape=(signal_length,)))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(2, input_shape= (100,) ,activation='softmax'))


    #
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=250, batch_size=50)

    file= open("model.pk", "wb")
    pk.dump(model, file)
    file.close()

    score = model.evaluate(x_test, y_test, batch_size= num_samples, verbose= 0)
    print(model.metrics_names[0],score[0])
    print(model.metrics_names[1], score[1])
    return model


def predict(model, to_predict):
    predictions=  model.predict(np.array([i for i in to_predict]))
    #print(predictions)
    print("output: \n")
    for i in predictions:
        #print(i)
        if i[0]>=i[1]:
            print([1,0])
        else:
            print([0,1])
    #print([1 for i in predictions if i>= .5 else 0])

"""
num_samples= 600
x_set, to_predict= six_signals(num_samples)
y_set= np.array([[1,0] for x in range(int(num_samples/2))]+ [[0,1] for x in range(int(num_samples/2))])
model = classify(x_set, y_set, num_samples)
predict(model, to_predict)
"""

x_set=[]
y_set=[]

script_dir = os.path.realpath(__file__) #<-- absolute dir the script is in

def splice(num_samples):
    x_set= []
    y_set= []
    for status in "01": #0 is good, 1 is bad
        for melon in "ab":
            filename= melon*2+status+".wav"
            print(filename)
            audio_signals= process(filename, dual_source=True)
            for sig in audio_signals:
                x_set.append(sig)
            if status == "0":
                #CHANGE= END SECOND -10
                for i in range(99):
                    y_set.append([1,0])
            else:
                for i in range(99):
                    y_set.append([0,1])
    file= open("data.pk", "wb")
    pk.dump([x_set, y_set], file)
    file.close()

num_samples= 4

splice(num_samples)
file= open("data.pk","rb")
x_set, y_set= pk.load(file)
file.close()
x_set= np.array(x_set)
print(x_set)
print("x_set shape: ", x_set.shape)

x_set= np.array(x_set)
y_set= np.array(y_set)
#CHANGE BASED ON CLIP SAMPLE SKIP
model= classify(x_set, y_set, num_samples, signal_length= 320)
#
#print(x_set.shape)
#print(y_set.shape)

#test_signal= process("watermeloncracktest.wav", dual_source=True)
#print(test_signal.shape)
#trial_classify(test_signal)

"""
each wav file (except ss0.wav) is 2 minutes, with 100 knocks. 00:10 to 01:49.
"""



# for status in ["1","2"]: # 1 = good, 2 = bad
#     for melon in ["A","B","C"]:
#         for clip in range(10, 107):
#             rel_path = "splices\\"+melon+status+"\\"+"clip"+str(clip)+".wav"
#             abs_file_path = os.path.join(script_dir[:-16], rel_path)
#             audio_signal= process_clip(abs_file_path)
#             x_set.append(audio_signal)
#             if status== "1": y_set.append([1,0])
#             else: y_set.append([0,1])

def old_six_signals():
    audio_signals=[]
    to_predict=[]
    for x in range(1, 4):
        to_predict.append(process(str(x)+".wav"))
        for y in range(100):
            audio_signals.append(process(str(x)+".wav"))
    for x in range(1, 4):
        for y in range(100):
            audio_signals.append(process("S"+str(x)+".wav"))
    return audio_signals, to_predict

def six_signals(num_samples):
    training_data= []
    to_predict= []
    for x in range(1, 4):
        to_predict.append(process(str(x)+".wav"))
        for y in range(100):
            training_data.append(process(str(x)+".wav"))

    for x in range(1,4):
        to_predict.append(process("S"+str(x)+".wav"))
        for y in range(100):
            training_data.append(process("S"+str(x)+".wav"))
    return training_data, to_predict
