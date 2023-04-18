import socket
import tkinter as tk
from tkinter import ttk
import sv_ttk
from Utils import Params
from ClientVideo import ClientVideo
from PIL import Image, ImageTk
import cv2

CamID = 0


class ClientApp:
    def __init__(self, root_window, windowName):
        self.photo = None
        self.root_window = root_window
        self.root_window.title(windowName)
        self.root_window.geometry("%dx%d" % (Params.VID_W + 30, Params.VID_H + 90))

        sv_ttk.set_theme('dark')  # setting up svttk theme

        self.canvas = tk.Canvas(root_window, width=Params.VID_W, height=Params.VID_H)
        self.canvas.configure(bg='black')
        self.canvas.place(relx=0.5, rely=0.5, anchor='center')

        self.calib_text = tk.StringVar()
        self.calib_text.set("Calibrate my camera")
        self.calib_btn = ttk.Button(self.root_window, textvariable=self.calib_text, width=20,
                                    command=self.calib_cam, style="Accent.TButton")

        self.calib_btn.place(relx=0.5, rely=0.01, anchor='n')

        self.canvas.bind("<Button-1>", self.handle_user_left_click)
        self.canvas.bind("<Button-3>", self.handle_user_right_click)

        self.cam_entry_frame = ttk.Frame(self.root_window, width=Params.VID_W, height=20)

        self.cam_entry_frame.pack(side="bottom")

        self.cam_id_entry = ttk.Entry(self.cam_entry_frame, width=5)
        self.cam_id_select_btn = ttk.Button(self.cam_entry_frame, text='Select Camera', width=20,
                                            command=self.config_cam)

        self.thread_clientVid = ClientVideo()

        self.root_window.protocol("WM_DELETE_WINDOW", lambda: self.close(self.root_window))

        # start ip window
        self.popup_ip_window = PopupWindow(self)
        self.popup_ip_window.ip_window.wait_window()

        if self.popup_ip_window.connected:
            # if the popup closed when connection was established

            # start client video thread
            self.thread_clientVid.setDaemon(True)
            self.thread_clientVid.start()

            self.calib_cam()

            self.play_selfie_video()

            self.root_window.mainloop()

    def handle_user_right_click(self, event):
        if self.thread_clientVid.calib_flag:
            self.thread_clientVid.mouse_location = None

    def handle_user_left_click(self, event):
        if self.thread_clientVid.calib_flag:
            x = event.x
            y = event.y
            self.thread_clientVid.mouse_location = x, y
            print("Mouse clicked at x =", x, "y =", y)

    def calib_cam(self):
        cab = self.thread_clientVid.calib_flag
        self.thread_clientVid.toggle_calib()
        if cab:  # if in calib switch to normal
            self.cam_id_entry.grid_forget()
            self.cam_id_select_btn.grid_forget()
            self.calib_text.set("Calibrate my camera")
        else:  # if in normal switch to calib
            self.cam_id_entry.grid(row=0, column=0, padx=5, pady=5)
            self.cam_id_select_btn.grid(row=0, column=1, padx=5, pady=5)
            self.calib_text.set("Finish calibration")

    def config_cam(self):
        text = self.cam_id_entry.get()
        try:
            cam_id = int(text)
        except ValueError:
            print("Invalid camera id")
            return
        try:
            self.thread_clientVid.set_cam(cam_id)
        except IOError as e:
            print("Error setting camera: ", e)

    def play_selfie_video(self):
        img = self.thread_clientVid.get_Queue()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buff = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=buff)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(10, self.play_selfie_video)

    def close(self, window):
        if self.thread_clientVid.is_alive():
            # check if the thread_clientVid is run yet
            self.thread_clientVid.close()
        window.destroy()


class PopupWindow:
    def __init__(self, caller):
        self.connected = False
        self.caller = caller
        self.root_window = caller.root_window
        self.ip_window = tk.Toplevel(self.root_window)
        self.ip_window.title("Enter IP")
        self.ip_window.geometry("400x100")
        self.ip_window.grab_set()

        self.ip_window.attributes('-fullscreen', False)
        self.ip_window.attributes("-topmost", True)  # <-- Add this line
        # self.ip_window.configure(bg='white')

        self.ip_entry = ttk.Entry(self.ip_window, width=30)
        self.ip_entry.focus_set()
        self.button = ttk.Button(self.ip_window, text="Enter", command=self.set_IP, style="Accent.TButton")
        self.ip_window.bind('<Return>', self.on_enter_event)
        self.error_label = tk.Label(self.ip_window, text="Failed to connect", fg="red")

        self.ip_entry.pack(side="top", expand=True)
        self.button.pack(side="top", expand=True)

        self.ip_window.protocol("WM_DELETE_WINDOW", self.close_all)

    def popup_close(self):
        self.ip_window.grab_release()
        self.ip_window.destroy()
        self.root_window.lift()

    def close_all(self):
        self.popup_close()
        self.caller.close(self.root_window)

    def on_enter_event(self, event):
        self.set_IP()

    def set_IP(self):
        self.error_label.pack_forget()
        ip = str(self.ip_entry.get())
        if ip == "":
            return
        try:
            self.caller.thread_clientVid.set_connection(ip)
            self.connected = True
            self.popup_close()
        except socket.timeout as e:
            self.error_label.pack(side="top", expand=True)
            print("Connection time out: ", e)
        except socket.error as e:
            self.error_label.pack(side="top", expand=True)
            print("Error setting IP: ", e)


if __name__ == '__main__':
    ClientApp(tk.Tk(), "Meeting")
