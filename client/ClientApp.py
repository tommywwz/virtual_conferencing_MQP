import socket
import tkinter as tk
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
        self.root_window.geometry("%dx%d" % (Params.VID_W + 30, Params.VID_H + 30))

        self.canvas = tk.Canvas(root_window, width=Params.VID_W, height=Params.VID_H)
        self.canvas.configure(bg='black')
        self.canvas.place(relx=0.5, rely=0.5, anchor='center')

        self.calib_btn = tk.Button(self.root_window, text='Calibrate my camera', width=20,
                                   height=2, command=lambda: self.clientVid.toggle_calib())
        self.calib_btn.pack()

        self.root_window.protocol("WM_DELETE_WINDOW", lambda: self.close(self.root_window))

        sv_ttk.set_theme('dark')  # setting up svttk theme

        self.canvas.bind("<Button-1>", self.handle_user_left_click)
        self.canvas.bind('<Button-3>', self.handle_user_right_click)

        self.clientVid = ClientVideo(CamID)

        self.popup_ip_window = PopupWindow(self)
        self.popup_ip_window.ip_window.wait_window()

        if self.popup_ip_window.connected:
            # init client video thread
            # start client video thread
            self.clientVid.start()
            self.play_selfie_video()

            self.root_window.mainloop()

    def handle_user_right_click(self, event):
        self.clientVid.mouse_location = None

    def handle_user_left_click(self, event):
        x = event.x
        y = event.y
        self.clientVid.mouse_location = x, y
        print("Mouse clicked at x =", x, "y =", y)

    def play_selfie_video(self):
        img = self.clientVid.get_Queue()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buff = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=buff)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(15, self.play_selfie_video)

    def close(self, window):
        if self.popup_ip_window.connected:
            self.clientVid.close()
        window.destroy()


class PopupWindow:
    def __init__(self, caller):
        self.connected = False
        self.caller = caller
        self.root_window = caller.root_window
        self.ip_window = tk.Toplevel(self.root_window)
        self.ip_window.title("Enter IP")
        self.ip_window.geometry("400x50")
        self.ip_window.grab_set()

        self.ip_window.attributes('-fullscreen', False)
        self.ip_window.attributes("-topmost", True)  # <-- Add this line
        self.ip_window.configure(bg='white')

        self.ip_entry = tk.Entry(self.ip_window, width=30)
        self.ip_entry.pack()
        self.ip_entry.focus_set()

        self.button = tk.Button(self.ip_window, text="Enter", command=self.set_IP)
        self.button.pack()
        # self.ip_entry.bind('<Return>', self.handle_ip_entry)

        self.ip_window.protocol("WM_DELETE_WINDOW", self.close_all)

    def popup_close(self):
        self.ip_window.grab_release()
        self.ip_window.destroy()
        self.root_window.lift()

    def close_all(self):
        self.popup_close()
        self.caller.close(self.root_window)

    def set_IP(self):
        try:
            self.caller.clientVid.set_connection(str(self.ip_entry.get()))
            self.connected = True
            self.popup_close()
        except socket.error:
            print("Error setting IP")


if __name__ == '__main__':
    ClientApp(tk.Tk(), "Meeting")
