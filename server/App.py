import tkinter as tk
import cv2
from Utils import Params
import PIL.Image, PIL.ImageTk
import os
import video_joint

current_path = os.path.dirname(__file__)
root_path = os.path.split(current_path)[0]


# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, root_window, window_title, video_source=0):
        self.foto = None
        self.photo = None
        self.root_window = root_window
        self.window_title = window_title
        self.video_source = video_source
        self.root_window.title(self.window_title)
        self.calib_window_closed = True

        self.canvas = tk.Canvas(root_window, width=Params.BG_W, height=Params.BG_H)
        self.canvas.pack()

        self.VI = video_joint.VideoInterface()
        self.VI.run()

        # self.show_vid()

        self.delay = 15
        self.root_play_video()

        # self.root_window.bind('t', self.new_popup_window)
        self.calib_btn = tk.Button(self.root_window, text='Calibrate my camera', width=20,
                                   height=2, bd='3', command=self.new_popup_window)
        self.calib_btn.pack()

        self.root_window.protocol("WM_DELETE_WINDOW", lambda: self.close(self.root_window))

        self.root_window.mainloop()

    def root_play_video(self):
        frame = self.VI.Q_FrameForDisplay.get()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(self.delay, self.root_play_video)

    def close(self, window):
        self.VI.stop()
        window.destroy()

    def new_popup_window(self):
        if self.calib_window_closed:  # when a popup is running: calib_window_closed == False, exit the function
            new_window = tk.Toplevel(self.root_window)
            new_window.title("Calibration")

            self.calib_window_closed = False
            self.VI.calib = True

            canvas = tk.Canvas(new_window, width=Params.VID_W,
                               height=Params.VID_H)
            canvas.pack()

            btn = tk.Button(new_window, text='Looks Good!', width=20,
                            height=2, bd='3', command=lambda: close_pop_window(new_window))
            btn.pack()

            # setup closing protocol
            new_window.protocol("WM_DELETE_WINDOW", lambda: close_pop_window(new_window))

            def pop_play_video():
                # try:
                cam = self.VI.Q_userFrame.get()
                cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)

                self.foto = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cam))
                canvas.create_image(0, 0, image=self.foto, anchor=tk.NW)

                new_window.after(self.delay, pop_play_video)

            pop_play_video()

            def close_pop_window(window):
                # set the flag to indicate that the window has been closed
                self.calib_window_closed = True
                self.VI.calib = False
                window.destroy()


App(tk.Tk(), "Meeting")
