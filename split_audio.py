import ffmpeg
from pydub import AudioSegment

filename="cc1.wav"
size=216
length=1000

for i in range(1, int(size/2)):
    newAudio = AudioSegment.from_wav(filename)
    newAudio = newAudio[length*i-length/2:length*i+length-length/2]
    newAudio.export("splices/C2/clip%s.wav" % i)
