# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct
import threading
import pyshine as ps
from wheels import edge_detection
from wheels.Frame import Frame

IF_QUIT = False

buff_4K = 4*1024

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(3, 640)  # width
vid.set(4, 360)  # height
PORT = 9999
HOST_IP = '192.168.1.3'  # paste your server ip address here

ed = edge_detection.EdgeDetection()
camID = socket.gethostbyname(socket.gethostname())
print(camID)
frameClass = Frame(camID)


def video_stream():
    # create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((HOST_IP, PORT))  # a tuple

    while vid.isOpened():
        success, frame = vid.read()
        if success:
            rsz_image = frame.copy()  # manipulate raw frame here
            edge = ed.process_frame(rsz_image, threshold=100)
            a, b = edge
            if a is not None and b is not None:
                h, w, c = rsz_image.shape
                cv2.line(rsz_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
                # cv2.imshow("test"+str(camID), resized_frame)
            else:
                edge = (0, 0)
            frameClass.updateFrame(image=rsz_image, edge_line=edge)  # update edge information

            pickled_frame = pickle.dumps(frameClass)

            # data length followed by serialized frame object
            msg = struct.pack("Q", len(pickled_frame)) + pickled_frame
            client_socket.sendall(msg)

            cv2.imshow('Transitting Video', rsz_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                global IF_QUIT
                IF_QUIT = True
                client_socket.close()
                break


def audio_stream():
    # https://pyshine.com/Socket-Programming-send-receive-live-audio/
    # CONSIDER TO USE WebRTC
    # https://www.100ms.live/blog/webrtc-python-react-app
    audio, context = ps.audioCapture(mode='send')

    # create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    client_socket.connect((HOST_IP, PORT-1))  # a tuple

    while True:
        frame = audio.get()

        a = pickle.dumps(frame)
        message = struct.pack("Q", len(a)) + a
        client_socket.sendall(message)

        if IF_QUIT:
            client_socket.close()
            break


thread0 = threading.Thread(target=video_stream)
thread1 = threading.Thread(target=audio_stream)
thread0.start()
thread1.start()
print("starting threads")
thread0.join()
thread1.join()

print("All threads are terminated")

