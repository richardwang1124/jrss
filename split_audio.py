import ffmpeg
from pydub import AudioSegment

filename="220.wav"
size=220
length=2000

for i in range(0, int(size/2)):
    newAudio = AudioSegment.from_wav(filename)
    newAudio = newAudio[length*i:length*i+2000]
    newAudio.export("splices/A1/clip%s.wav" % i)
