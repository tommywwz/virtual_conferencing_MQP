import threading
import tkinter as tk
from tkinter import ttk
import cv2
from Utils import Params, Tools, CamManagement
from PIL import Image, ImageTk
import os
from server import video_joint
import time

current_path = os.path.dirname(__file__)
root_path = os.path.split(current_path)[0]


# source: https://solarianprogrammer.com/2018/04/21/python-opencv-show-video-tkinter-window/
class ServerApp:
    def __init__(self, root_window, window_title):
        self.foto = None  # holds the pop-up window image
        self.photo = None  # holds the main window image
        self.root_window = root_window
        self.root_window.title(window_title)
        self.root_window.geometry("%dx%d" % (Params.BG_W + 30, Params.BG_H + 30))
        self.root_window.attributes('-fullscreen', True)

        self.calib_window_closed = True

        width = self.root_window.winfo_screenwidth()
        height = self.root_window.winfo_screenheight()
        self.width_cam = int(width / 2)
        self.height_cam = int(height / 2)
        # self.root_window.configure(bg='black')

        self.canvas = tk.Canvas(self.root_window, width=self.width_cam, height=self.height_cam)
        self.canvas.configure(bg='black')
        self.canvas.place(relx=0.5, rely=0.5, anchor='center')

        self.host_ip_text = tk.StringVar()
        self.host_ip_text.set("Host IP: " + Tools.get_host_ip())
        self.host_ip_label = ttk.Label(self.root_window, textvariable=self.host_ip_text, width=20,
                                       style="Accent.TLabel")

        self.VJ = video_joint.VideoJoint()
        self.VJ.run()

        # self.show_vid()

        # self.new_popup_window()  # open the popup window at the start

        # self.root_window.bind('t', self.new_popup_window)
        self.calib_btn = ttk.Button(self.root_window, text='Calibrate my camera', width=20,
                                    command=self.start_popup_window, style='TButton')

        self.exit_btn = tk.Button(self.root_window, text='\u274C',
                                  bd='0', width=7, height=3,
                                  command=lambda: self.close_main_window())

        self.calib_btn.place(relx=0.5, rely=0.01, anchor='n')
        self.exit_btn.place(relx=1.0, rely=0, anchor='ne')
        self.host_ip_label.place(relx=0, rely=1, anchor='sw')

        self.root_window.protocol("WM_DELETE_WINDOW", lambda: self.close_main_window())

        self.main_delay = 15
        self.root_play_video()

        if __name__ == '__main__':
            self.root_window.mainloop()

    def root_play_video(self):
        start_time = time.time()

        frame = self.VJ.Q_FrameForDisplay.get()

        if self.calib_window_closed:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.width_cam, self.height_cam))

            buff = Image.fromarray(frame)
            self.photo = ImageTk.PhotoImage(image=buff)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(self.main_delay, self.root_play_video)

    def close_main_window(self):
        self.VJ.stop()
        self.root_window.destroy()

    def start_popup_window(self):
        pop_up_thread = PopUpWindow(self)
        pop_up_thread.setDaemon(True)
        pop_up_thread.start()


class PopUpWindow(threading.Thread):
    def __init__(self, root):
        threading.Thread.__init__(self)
        self.root = root
        self.root.calib_btn.configure(state='disabled')
        self.new_window = tk.Toplevel(self.root.root_window)
        self.new_window.title("Calibration")

        self.root.calib_window_closed = False
        self.CamMan_singleton = CamManagement.CamManagement()
        self.CamMan_singleton.calib = True

        self.canvas = tk.Canvas(self.new_window, width=Params.VID_W,
                                height=Params.VID_H)
        self.left_padding_canvas = tk.Canvas(self.new_window, width=1,
                                             height=Params.VID_H)
        self.right_padding_canvas = tk.Canvas(self.new_window, width=1,
                                              height=Params.VID_H)

        self.canvas.bind("<Button-1>", self.handle_user_left_click)
        self.canvas.bind('<Button-3>', self.handle_user_right_click)

        self.btn = ttk.Button(self.new_window, text='Finish Calibration', width=20,
                              command=lambda: self.close_pop_window(), style='Accent.TButton')

        self.cam_entry = ttk.Entry(self.new_window, width=5)
        self.cam_select_btn = ttk.Button(self.new_window, text='Select Camera', width=20,
                                         command=lambda: self.select_camera())

        self.canvas.grid(row=0, column=0, columnspan=3, padx=10, pady=2)

        self.btn.grid(row=1, column=0, padx=4)
        self.cam_select_btn.grid(row=1, column=1, padx=4, pady=5)
        self.cam_entry.grid(row=1, column=2, padx=4, pady=5)
        # setup closing protocol
        self.new_window.protocol("WM_DELETE_WINDOW", lambda: self.close_pop_window())
        self.pop_delay = 15
        self.video_running = True
        self.photo = None
        self.play_video()

    def handle_user_right_click(self, event):
        self.root.VJ.mouse_location_FE = None

    def handle_user_left_click(self, event):
        x = event.x
        y = event.y
        self.root.VJ.mouse_location_FE = x, y
        print("Mouse clicked at x =", x, "y =", y)

    def select_camera(self):
        cam_id = self.cam_entry.get()
        if cam_id.isdigit():
            self.root.VJ.update_user_cam_FE(int(cam_id))
            print("Camera selected: ", cam_id)
        else:
            print("Camera selection failed. Please enter a number.")

    def play_video(self):

        frame = self.root.VJ.Q_userFrame.get()

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
        while not self.root.VJ.Q_userFrame.empty():
            item = self.root.VJ.Q_userFrame.get()
        self.CamMan_singleton.calib = False
        self.root.calib_btn.configure(state='normal')
        self.root.root_window.focus_set()
        self.new_window.destroy()


if __name__ == '__main__':
    ServerApp(tk.Tk(), "Meeting")
