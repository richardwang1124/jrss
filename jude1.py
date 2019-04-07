import pyaudio
import wave
import sys

form_1 = pyaudio.paInt32 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 32000 # 44.1kHz sampling rate
chunk = int(sys.argv[3]) # 2^12 samples for buffer
record_secs = int(sys.argv[2]) # seconds to record
dev_index = 3 # device index found by p.get_device_info_by_index(ii)

#print(sys.argv)
fn = sys.argv[1]+".wav"
wav_output_filename = fn # name of .wav file

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
#print("recording")
#print(sys.argv)

frames = []

# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
    data = stream.read(chunk)
    frames.append(data)

print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

# save the audio frames as .wav file
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()

print("DONE")
print("FILENAME: "+fn)

