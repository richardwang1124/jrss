import serial

file=open("audio_amplitudes/text.txt", 'w')

ser = serial.Serial(port='/dev/tty96B0', baudrate=115200)
size=0
while size < 500:
	file.write(str(ser.readline())+'\n')
	size +=1
file.write(str(size))
