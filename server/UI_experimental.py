import tkinter as tk
import cv2
import Params
import PIL.Image, PIL.ImageTk
from assets import vid

# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window_title = window_title
        self.video_source = video_source

        self.canvas = tk.Canvas(window, width=Params.BG_W, height=Params.BG_H)
        self.canvas.pack()

        self.vid = cv2.VideoCapture("demo2.mp4")

        # self.show_vid()

        self.delay = 15
        self.update()

        self.window.mainloop()

    # def show_vid(self):
    #     if self.vid.isOpened():
    #         ret, frame = self.vid.read()
    #         # Convert the frame to a Tkinter PhotoImage object
    #         if ret:
    #             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #             img = Pil_imageTk.PhotoImage(image=Pil_image.fromarray(frame))
    #
    #             # Show the frame on the canvas
    #             self.canvas.create_image(0, 0, image=img, anchor=tk.NW)
    #
    #             # Repeat the process after a delay of 30ms
    #             self.window.after(30, self.show_video)

    def update(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)


App(tk.Tk(), "Tkinter and OpenCV")
