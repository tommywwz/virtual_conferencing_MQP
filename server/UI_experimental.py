import tkinter as tk
import cv2


# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window_title = window_title

        self.window.mainloop()


App(tk.Tk(), "Tkinter and OpenCV")
