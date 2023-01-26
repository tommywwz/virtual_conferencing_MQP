# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct
import threading
import pyaudio
from wheels import edge_detection
from wheels.Frame import Frame

IF_QUIT = False

buff_4K = 4*1024

cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height
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

    while cam.isOpened():
        success, frame = cam.read()
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
                cam.release()
                client_socket.close()
                break


def audio_stream():
    # Create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Define the server's address and port
    server_address = ("127.0.0.1", 12345)

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open a microphone input stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=44100,
                    input=True,
                    frames_per_buffer=1024)

    # Create a unique identifier for the client
    client_id = b"1"

    # Initialize the packet number
    packet_num = 0

    # Continuously send audio packets to the server
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(1024)

        # Pack the data with the packet number and client id
        packet = client_id + struct.pack("!H", packet_num) + audio_data

        # Send the packet to the server
        s.sendto(packet, server_address)

        # Increment the packet number
        packet_num += 1

        if IF_QUIT:
            break

    # Close the audio input stream
    stream.stop_stream()
    stream.close()
    p.terminate()


thread0 = threading.Thread(target=video_stream)
thread1 = threading.Thread(target=audio_stream)
thread0.start()
thread1.start()
print("starting threads")
thread0.join()
thread1.join()

print("All threads are terminated")
exit(0)
