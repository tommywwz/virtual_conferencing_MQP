# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct
import threading
from wheels import edge_detection
from wheels.Frame import Frame

IF_QUIT = False


cam = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam.set(3, 640)  # width
cam.set(4, 360)  # height
PORT = 9999
HOST_IP = '192.168.1.3'  # paste your server ip address here
socket_addr = (HOST_IP, PORT+1)

if __name__ == '__main__':
    ed = edge_detection.EdgeDetection()
    camID = socket.gethostbyname(socket.gethostname())
    print(camID)
    frameClass = Frame(camID)
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        ret, frame = cam.read()
        frame = cv2.resize(frame, (640, 480))
        data = frame.tostring()
        size = str(len(data)).zfill(10)
        for i in range(0, len(data), 512):
            client.sendto((size + data[i:i + 512]).encode('utf-8'), socket_addr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    client.close()
    cv2.destroyAllWindows()

    # def video_stream():
    #     # create socket
    #     client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #
    #     client_socket.connect((HOST_IP, PORT))  # a tuple
    #
    #     while cam.isOpened():
    #         success, frame = cam.read()
    #         if success:
    #             rsz_image = frame.copy()  # manipulate raw frame here
    #             edge = ed.process_frame(rsz_image, threshold=100)
    #             a, b = edge
    #             if a is not None and b is not None:
    #                 h, w, c = rsz_image.shape
    #                 cv2.line(rsz_image, (0, round(b)), (w, round((w * a + b))), (0, 255, 0), 2)
    #                 # cv2.imshow("test"+str(camID), resized_frame)
    #             else:
    #                 edge = (0, 0)
    #             frameClass.updateFrame(image=rsz_image, edge_line=edge)  # update edge information
    #
    #             pickled_frame = pickle.dumps(frameClass)
    #
    #             # data length followed by serialized frame object
    #             msg = struct.pack("Q", len(pickled_frame)) + pickled_frame
    #             client_socket.sendall(msg)
    #
    #             cv2.imshow('Transitting Video', rsz_image)
    #             key = cv2.waitKey(1) & 0xFF
    #             if key == ord('q'):
    #                 global IF_QUIT
    #                 IF_QUIT = True
    #                 cam.release()
    #                 client_socket.close()
    #                 break


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

    # thread0 = threading.Thread(target=video_stream)
    # thread1 = threading.Thread(target=audio_stream)
    # thread0.start()
    # thread1.start()
    # print("starting threads")
    # thread0.join()
    # thread1.join()
    # print("All threads are terminated")
    exit(0)
