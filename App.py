import tkinter as tk
from tkinter import ttk, font
import sv_ttk
from Utils import *
from server import ServerApp
from client import ClientApp


class App:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Main')
        self.root.geometry('300x80')
        self.root.resizable(False, False)
        self.root.attributes('-fullscreen', False)
        # Just simply import the azure.tcl file
        self.root.tk.call("source", "azure.tcl")

        self.root.tk.call("set_theme", "dark")

        # self.root.configure(bg='black')
        self.root.focus_force()
        self.fl = tk.Frame(self.root)
        self.fr = tk.Frame(self.root)
        self.fl.grid(row=0, column=0, sticky="nsew")
        self.fr.grid(row=0, column=1, sticky="nsew")

        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.server_btn = ttk.Button(self.fl, text='Start as Server', command=self.open_server)
        self.client_btn = ttk.Button(self.fr, text='Start as Client', command=self.open_client)
        self.server_btn.place(relx=0.5, rely=0.5, anchor='center')
        self.client_btn.place(relx=0.5, rely=0.5, anchor='center')

        self.root.protocol("WM_DELETE_WINDOW", self.close_main_window)
        # sv_ttk.set_theme('dark')
        self.root.mainloop()

    def open_server(self):
        self.root.withdraw()
        ServerApp.ServerApp(tk.Toplevel(self.root), "Meeting")
        self.root.deiconify()
        self.root.focus_set()

    def open_client(self):
        self.root.withdraw()
        ClientApp.ClientApp(tk.Toplevel(self.root), "Meeting")
        self.root.deiconify()
        self.root.focus_set()

    def close_main_window(self):
        self.root.destroy()


if __name__ == '__main__':
    App()
