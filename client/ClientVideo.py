# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct
import threading
import queue
from Utils import edge_detection, Params
from Utils.Frame import Frame
from Utils.AutoResize import AutoResize


buff_4K = 4 * 1024
PORT = 9999
HOST_IP = '192.168.1.3'  # paste your server ip address here


class ClientVideo(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        self.exit_flag = False
        self.client_socket = None
        self.Q_selfie = queue.Queue(maxsize=3)

        self.cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.cam.set(3, 640)  # width
        self.cam.set(4, 360)  # height

        camID = socket.gethostbyname(socket.gethostname())
        print("HOST NAME: ", camID)
        self.frameClass = Frame(camID)

    def run(self):
        # create socket
        ed = edge_detection.EdgeDetection()
        client_auto_resize = AutoResize()

        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((HOST_IP, PORT))  # a tuple

        while self.cam.isOpened():
            # todo need calib there (resize & edge)
            success, frame = self.cam.read()
            break

        while self.cam.isOpened():
            success, frame = self.cam.read()
            if success:
                rsz_image = cv2.rotate(frame.copy(), cv2.ROTATE_90_CLOCKWISE)  # manipulate raw frame here
                edge = ed.process_frame(rsz_image, threshold=100)
                a, b = edge
                if a is None or b is None:  # if edge information is not valid, give it a default value
                    edge = (0, 0)

                # start of image resizing
                ratio = client_auto_resize.resize(rsz_image, 100)
                adjust_w = round(Params.VID_W * ratio)
                adjust_h = round(Params.VID_H * ratio)
                new_shape = (adjust_w, adjust_h)
                if ratio > 1:
                    rsz_image = cv2.resize(rsz_image, new_shape, interpolation=cv2.INTER_LINEAR)
                    ah, aw = rsz_image.shape[:2]
                    dn = round(ah * 0.5 + Params.VID_H * 0.5)
                    up = round(ah * 0.5 - Params.VID_H * 0.5)
                    lt = round(aw * 0.5 - Params.VID_W * 0.5)
                    rt = round(aw * 0.5 + Params.VID_W * 0.5)
                    rsz_image = rsz_image[up:dn, lt:rt]
                else:
                    rsz_image = cv2.resize(rsz_image, new_shape, interpolation=cv2.INTER_AREA)
                # end of image resizing

                self.frameClass.updateFrame(image=rsz_image, edge_line=edge)  # update edge information

                pickled_frame = pickle.dumps(self.frameClass)

                # data length followed by serialized frame object
                msg = struct.pack("Q", len(pickled_frame)) + pickled_frame
                self.client_socket.sendall(msg)

                self.Q_selfie.put(rsz_image)
                # cv2.imshow('Transitting Video', rsz_image)
                if self.exit_flag:
                    break

    def get_Queue(self):
        return self.Q_selfie.get()

    def dump_Queue(self):
        if not self.Q_selfie.empty():
            while not self.Q_selfie.empty():
                item = self.Q_selfie.get()
                print("dequeued one item")
        else:
            print("The queue is empty.")

    def close(self):
        self.exit_flag = True
        self.cam.release()
        self.client_socket.close()


# # Create a socket object
# s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#
# # Set the server's IP address and port
# server_address = (HOST_IP, PORT-1)
#
# # Get the client's unique identifier
# client_id = input("Enter a unique identifier for the client: ").encode()
#
# # Define the audio settings
# fs = 44100 # Sample rate
# channels = 2 # Number of channels
#
# # Continuously record audio and send it to the server
# packet_number = 0


# def audio_callback(indata, frames, time, status):
#     global packet_number
#     audio = indata.copy()
#     # Send the audio to the server
#     s.sendto(client_id + b":" + str(packet_number).encode() + b":" + audio.tobytes(), server_address)
#     packet_number += 1
#     # Wait for an acknowledgement from the server
#     s.recvfrom(1024)
#
#
# def audio_stream():
#     # Open the microphone stream
#     stream = sd.InputStream(callback=audio_callback, samplerate=fs, channels=channels)
#
#     stream.start()
#
#     while True:
#         if IF_QUIT:
#             break
#     # Close the socket
#     s.close()
if __name__ == "__main__":
    thread0 = ClientVideo()
    # thread1 = threading.Thread(target=audio_stream)
    thread0.start()
    # thread1.start()
    print("starting threads")
    thread0.join()
    # thread1.join()

    print("All threads are terminated")
    exit(0)
