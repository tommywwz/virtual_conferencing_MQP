import threading
import tkinter as tk
import sv_ttk
import cv2
from Utils import Params
from PIL import Image, ImageTk
import os
import video_joint
import time

current_path = os.path.dirname(__file__)
root_path = os.path.split(current_path)[0]


# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class App:
    def __init__(self, root_window, window_title, video_source=0):
        self.foto = None  # holds the pop-up window image
        self.photo = None  # holds the main window image
        self.root_window = root_window
        self.video_source = video_source
        self.root_window.title(window_title)
        self.calib_window_closed = True

        width = self.root_window.winfo_screenwidth()
        height = self.root_window.winfo_screenheight()
        self.width_cam = int(width / 2)
        self.height_cam = int(height / 2)
        self.root_window.geometry("%dx%d" % (Params.BG_W + 30, Params.BG_H + 30))
        self.root_window.attributes('-fullscreen', True)
        # self.root_window.configure(bg='black')

        self.canvas = tk.Canvas(root_window, width=self.width_cam, height=self.height_cam)
        self.canvas.configure(bg='black')
        self.canvas.place(relx=0.5, rely=0.5, anchor='center')

        self.VI = video_joint.VideoInterface()
        self.VI.run()

        # self.show_vid()

        # self.new_popup_window()  # open the popup window at the start

        # self.root_window.bind('t', self.new_popup_window)
        self.calib_btn = tk.Button(self.root_window, text='Calibrate my camera', width=20,
                                   height=2, command=self.start_popup_window)
        self.calib_btn.pack()

        self.exit_btn = tk.Button(self.root_window, text='\u274C',
                                  bd='0', width=7, height=3,
                                  command=lambda: self.close_main_window(self.root_window))
        self.exit_btn.place(relx=1.0, rely=0, anchor='ne')

        self.root_window.protocol("WM_DELETE_WINDOW", lambda: self.close_main_window(self.root_window))

        self.main_delay = 15
        self.root_play_video()

        sv_ttk.set_theme('dark')  # setting up svttk theme
        self.root_window.mainloop()

    def root_play_video(self):
        start_time = time.time()

        frame = self.VI.Q_FrameForDisplay.get()

        if self.calib_window_closed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width_cam, self.height_cam))

            duration = int(time.time() - start_time)
            if duration is not 0:
                fps = round(1.0 / duration)
                cv2.putText(frame,
                            text='FPS: ' + str(fps),
                            org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=.4, color=(0, 255, 255), thickness=1,
                            lineType=cv2.LINE_AA, bottomLeftOrigin=False)

            buff = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=buff)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(self.main_delay, self.root_play_video)

    def close_main_window(self, window):
        self.VI.stop()
        window.destroy()

    def start_popup_window(self):
        pop_up_thread = PopUpWindow(self)
        pop_up_thread.setDaemon(True)
        pop_up_thread.start()


class PopUpWindow(threading.Thread):
    def __init__(self, root):
        threading.Thread.__init__(self)
        self.root = root
        self.new_window = tk.Toplevel(self.root.root_window)
        self.new_window.title("Calibration")

        self.root.calib_window_closed = False
        self.root.VI.CamMan.calib = True

        self.canvas = tk.Canvas(self.new_window, width=Params.VID_W,
                                height=Params.VID_H)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.handle_user_left_click)
        self.canvas.bind('<Button-3>', self.handle_user_right_click)

        self.btn = tk.Button(self.new_window, text='Looks Good!', width=20,
                             height=2, bd='1', command=lambda: self.close_pop_window())
        self.btn.pack()

        # setup closing protocol
        self.new_window.protocol("WM_DELETE_WINDOW", lambda: self.close_pop_window())
        self.pop_delay = 15
        self.video_running = True
        self.photo = None
        self.play_video()

    def handle_user_right_click(self, event):
        self.root.VI.mouse_location_FE = None

    def handle_user_left_click(self, event):
        x = event.x
        y = event.y
        self.root.VI.mouse_location_FE = x, y
        print("Mouse clicked at x =", x, "y =", y)

    def play_video(self):

        frame = self.root.VI.Q_userFrame.get()
        print("getQ")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (Params.VID_W, Params.VID_H))

        buff = Image.fromarray(frame)
        self.photo = ImageTk.PhotoImage(image=buff)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.new_window.after(self.pop_delay, self.play_video)

    def close_pop_window(self):
        # set the flag to indicate that the window has been closed
        self.root.calib_window_closed = True
        print("-----------exiting calibration window------------")
        while not self.root.VI.Q_userFrame.empty():
            item = self.root.VI.Q_userFrame.get()
        self.root.VI.CamMan.calib = False
        self.new_window.destroy()


if __name__ == '__main__':
    App(tk.Tk(), "Meeting")
