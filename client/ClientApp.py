import tkinter as tk
import sv_ttk
from Utils import Params
import ClientVideo
from PIL import Image, ImageTk
import cv2

CamID = 3


class ClientApp:
    def __init__(self, root_window, windowName):
        self.photo = None
        # start client video thread
        self.clientVid = ClientVideo.ClientVideo(CamID)
        self.clientVid.start()

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

        self.delay = 15
        self.play_selfie_video()

        self.root_window.mainloop()

    def play_selfie_video(self):
        img = self.clientVid.get_Queue()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        buff = Image.fromarray(img)
        self.photo = ImageTk.PhotoImage(image=buff)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root_window.after(self.delay, self.play_selfie_video)

    def close(self, window):
        self.clientVid.close()
        window.destroy()


ClientApp(tk.Tk(), "Meeting")
