# Virtual_Conferencing_MQP

This program aims to improve the overall quality of the user experience (QoE) on existing online conference meeting platforms, such as Zoom or Teams. The objective of the system is to explore methods to increase immersion and comfort in virtual meetings while mitigating the negative effects of prolonged use, commonly referred to as "Zoom fatigue." The proposed solution consists of five functional modules: Multiple Target Windows Merging and Alignment, Background Removal, Vision Angle Tilting by Head Orientation, Auto Display Resizing, and Boundary Warning. Importantly, the system does not require additional hardware or virtual reality tools, as a standard built-in or USB webcam is sufficient.

## Demo Video
https://youtu.be/e3_ayWrDN_o

## Environment Configuration
This program was developed and tested on Windows 10 & 11, we strongly recommend using window to run this program.<br /><br />
**Known issue for Linux system:**
- the PyAudio package is not supported
### 1. Install Python 3.7
Python 3.7 can be installed from the link below:<br />
https://www.python.org/downloads/release/python-370/
### 2. load requirement
Make sure if pip is already installed:<br />
```bash
pip help
```
Install the required packages by using the following command
```bash
$ pip install -r requirements.txt
```

## Where to Start?

This project consist of two individual programs, the [Server App](./server/ServerApp.py), and the [Client App](./client/ClientApp.py). <br /><br />
There are the steps to set up and run the program:
1. Make sure all computers are connected in the same local network. One computer will be used as server and other computers will be client.
2. Start the application by running the [App.py](./App.py) file
3. For the server, click on the "Start Server" button, and for the client, click on the "Start Client" button.
4. In the client application, for connecting to server, enter the server IP address and click on the "Connect" button.