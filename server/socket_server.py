# reference: https://pyshine.com/Socket-programming-and-openc/
# require soundfile, sounddevice
import cv2
import pickle
import socket
import struct
import threading
import pyshine as ps
from wheels.Frame import Frame

PORT = 9999
buff_4K = 4 * 1024
HOST_IP = '192.168.1.3'


def video_client(client_socket, client_addr):
    data = b""
    payload_size = struct.calcsize("Q")
    windowName = str(client_addr)
    cv2.namedWindow(windowName)
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(buff_4K)  # 4K
            if not packet:
                break
            data += packet
        packed_msg_size = data[:payload_size]  # extracting the packet size information
        data = data[payload_size:]  # extract the img data from the rest of the packet
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            # keep loading the data until the entire data received
            data += client_socket.recv(buff_4K)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frameClass = pickle.loads(frame_data)
        frame = frameClass.img
        cv2.imshow(windowName, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            client_socket.close()
            cv2.destroyWindow(windowName)
            break


def audio_client(client_socket, client_addr):
    audio, context = ps.audioCapture(mode='get')
    data = b""
    payload_size = struct.calcsize("Q")
    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)  # 4K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frame = pickle.loads(frame_data)
        audio.put(frame)

    client_socket.close()


def video_manage():
    server_sock = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ips = socket.gethostbyname_ex(host_name)
    print('host ip: ' + HOST_IP)

    socket_addr = (HOST_IP, PORT)

    # socket bind
    server_sock.bind(socket_addr)

    # socket listen
    server_sock.listen(5)
    print('Listening at: ', socket_addr)

    while True:
        # waiting from client connection and create a thread for it
        client_socket, client_addr = server_sock.accept()
        print('GOT NEW VIDEO CONNECTION FROM: ', client_addr)
        if client_socket:
            newClientVidThread = threading.Thread(target=video_client, args=(client_socket, client_addr))
            newClientVidThread.start()
            print("starting thread for client:", client_addr)


def audio_manage():
    # https://pyshine.com/Socket-Programming-send-receive-live-audio/
    # CONSIDER TO USE WebRTC
    # https://www.100ms.live/blog/webrtc-python-react-app
    server_sock = socket.socket(socket.AF_INET,
                                socket.SOCK_STREAM)
    host_name = socket.gethostname()
    host_ips = socket.gethostbyname_ex(host_name)
    print('host ip: ' + HOST_IP)

    socket_addr = (HOST_IP, PORT-1)

    # socket bind
    server_sock.bind(socket_addr)

    # socket listen
    server_sock.listen(5)
    print('Listening at: ', socket_addr)

    while True:
        # waiting from client connection and create a thread for it
        client_socket, client_addr = server_sock.accept()
        print('GOT NEW AUDIO CONNECTION FROM: ', client_addr)
        if client_socket:
            newClientAudThread = threading.Thread(target=audio_client, args=(client_socket, client_addr))
            newClientAudThread.start()
            print("starting thread for client:", client_addr)


video_man = threading.Thread(target=video_manage)
audio_man = threading.Thread(target=audio_manage)
video_man.start()
print("starting thread0")

