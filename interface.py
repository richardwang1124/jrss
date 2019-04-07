import serial
import tkinter as tk
import time

ser = serial.Serial(port="/dev/cu.usbmodem14101", baudrate=9600)

def begin():
        ser.write(1)
        btn.configure(text="Stop Melon Analysis", command=end)
        
def end():
        ser.write(1)
        btn.configure(text="Initiate Melon Analysis", command=begin)
        

window=tk.Tk()
window.title("melon")
window.geometry('350x200')
good_melon=tk.PhotoImage(file="watermelon.gif")
bad_melon=tk.PhotoImage(file="smash.gif")
background_label = tk.Label(window, image=good_melon)
background_label.place(x=0, y=0, relwidth=1, relheight=1)
btn = tk.Button(window, text="Initiate Melon Analysis", width=350, command=begin)

btn.pack()
 
