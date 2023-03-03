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
There are several step to set up and run the program:
1. Make sure all computers are connected in the same local network. One computer will be used as server and other computers will be client.
2. Use ```ipconfig``` command to find the local network ip address of the server computer. 
3. Change the ```HOST_IP``` in [Params.py](./Utils/Params.py) to the server computer's IP address for all devices.
4. For the server side: 
   1. Set ```SERVER_CAM_ID``` in [video_joint.py](./server/video_joint.py) to the Camera ID the server side will be using (Camera ID normally starts from 0); to have the best experience, we recommend to rotate the camera 90 degree clockwise (from landscape to portrait).
   2. Run [ServerApp.py](./server/ServerApp.py) on the computer used as server.
5. For the client side:
   1. Set ```CamID``` in [ClientApp.py](./client/ClientApp.py) to the Camera ID the client side will be using (Camera ID normally starts from 0); to have the best experience, we recommend to rotate the camera 90 degree clockwise (from landscape to portrait).
   2. Run [ClientApp.py](./client/ClientApp.py) on client computers

