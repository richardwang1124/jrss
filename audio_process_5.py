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


def process(filename, signal_length= 100000, dual_source= False):
    audio_signals= [] #list of np arrays, outer length 100, inner length 0.5 seconds
    sample_rate, audio_signal = wavfile.read(filename)  # File assumed to be in the same directory
    #print("signal", audio_signal.shape)
    if dual_source:
         audio_signal= np.mean(audio_signal, axis= 1)
    #print(audio_signal.shape)

    N = audio_signal.shape[0]
    L = N / sample_rate
    #print("N: ", N)
    #print(f'Audio length: {L:.2f} seconds')
    #CHANGE
    for i in range(10,109):#(10,109):
        start= int((i-.12)*sample_rate)
        end= int(start+ sample_rate*.2)
        #CHANGE
        clip= audio_signal[start:end:3]
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
    #print(audio_signals)
    return audio_signals


def classify(x, y, num_samples, signal_length=2134):
    print("x input shape:", x.shape)
    #y = np.array([[1] for x in range(num_samples)])
    print("y ********************")
    #print(y)
    print("y shape", y.shape)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=.1)

    model = Sequential()
    model.add(Dense(50, activation='sigmoid', input_shape=(signal_length,)))
    #model.add(Dense(50, activation='sigmoid', input_shape=(signal_length,)))
    model.add(Dense(40, activation='relu'))
    model.add(Dropout(.4))
    #model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))


    #
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])
    model.fit(x, y, epochs=50, batch_size=25)

    file= open("model.pk", "wb")
    pk.dump(model, file)
    file.close()

    score = model.evaluate(x_test, y_test, batch_size= num_samples, verbose= 0)
    print(model.metrics_names[0],score[0])
    print(model.metrics_names[1], score[1])
    return model


def predict(model, to_predict, y_predict):
    predictions=  model.predict(np.array([i for i in to_predict]))
    #print(predictions)
    print("prediction accuracy: \n")
    sum = 0

    for i in range(len(to_predict)):

        if predictions[i][0] > 0.5 and y_predict[i][0] ==1:
            sum += 1
        if predictions[i][0] < 0.5 and y_predict[i][0] ==0:
            sum += 1
    print(sum/len(predictions))

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
        for melon in "abc":
            filename= melon*2+status+".wav"
            #print(filename)
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

num_samples= 6

splice(num_samples)
file= open("data.pk","rb")
x_set, y_set= pk.load(file)
file.close()
x_set= np.array(x_set)
#print(x_set)
print("x_set shape: ", x_set.shape)

x_set= np.array(x_set)
y_set= np.array(y_set)
#CHANGE BASED ON CLIP SAMPLE SKIP
model= classify(x_set, y_set, num_samples, signal_length= 2134)

#don't get this next line in the screenshot
x_train, x_predict, y_train, y_predict= train_test_split(x_set, y_set, test_size=.2)
predict(model, x_predict, y_predict)

"""
each wav file (except ss0.wav) is 2 minutes, with 100 knocks. 00:10 to 01:49.
"""
