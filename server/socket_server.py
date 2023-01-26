# reference: https://pyshine.com/Socket-programming-and-openc/
# require soundfile, sounddevice
import cv2
import pickle
import socket
import struct
import threading
import pyaudio
import wave
from wheels.Frame import Frame

PORT = 9999
buff_4K = 4 * 1024
HOST_IP = '192.168.1.3'

# Define the number of channels, sample rate, and sample width
channels = 2
sample_rate = 44100
sample_width = 2

clients = {}
# Create a lock to protect the clients dictionary
lock = threading.Lock()


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

        if not packed_msg_size:  # check if client has lost connection
            client_socket.close()
            cv2.destroyWindow(windowName)
            print("Lost connection from client:", client_addr)
            break

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


# def audio_client(client_socket, client_addr):
#     audio, context = ps.audioCapture(mode='get')
#     data = b""
#     payload_size = struct.calcsize("Q")
#     while True:
#         while len(data) < payload_size:
#             packet = client_socket.recv(4 * 1024)  # 4K
#             if not packet: break
#             data += packet
#         packed_msg_size = data[:payload_size]
#
#         if not packed_msg_size:  # check if client has lost connection
#             client_socket.close()
#             print("Lost connection from client:", client_addr)
#             break
#
#         data = data[payload_size:]
#         msg_size = struct.unpack("Q", packed_msg_size)[0]
#
#         while len(data) < msg_size:
#             data += client_socket.recv(4 * 1024)
#         frame_data = data[:msg_size]
#         data = data[msg_size:]
#         frame = pickle.loads(frame_data)
#         audio.put(frame)
#
#     client_socket.close()


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


def client_handler(data, addr):
    # """Handle incoming data from a client"""
    # Parse the client's unique identifier from the data
    client_id = data.split(b":")[0]
    # Acquire the lock
    lock.acquire()
    try:
        # Add the packet to the client's packet list
        clients[client_id].append(data[4:])
    finally:
        # Release the lock
        lock.release()


def audio_manage():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind the socket to a specific address and port
    server_sock.bind((HOST_IP, PORT-1))

    # Continuously listen for incoming packets
    while True:
        data, addr = server_sock.recvfrom(1024)
        # Spawn a new thread to handle the client
        t = threading.Thread(target=client_handler, args=(data, addr))
        t.start()


# Reassemble the audio stream from the packets
def reassemble_audio():
    while True:
        # Acquire the lock
        lock.acquire()
        try:
            # Check if all clients have sent packets
            if len(clients) == len(clients[client_id]):
                # Sort the packets by packet number
                for client_id in clients:
                    clients[client_id].sort(key=lambda x: x[0])
                # Reassemble the original audio stream
                audio_stream = b"".join([p[1] for client_id in clients for p in clients[client_id]])
                # Play the audio stream using PyAudio
                p = pyaudio.PyAudio()
                stream = p.open(format=p.get_format_from_width(sample_width),
                                channels=channels,
                                rate=sample_rate,
                                output=True)
                stream.start_stream()
                stream.write(audio_stream)
                stream.stop_stream()
                stream.close()
                p.terminate()
                clients.clear()
        finally:
            # Release the lock
            lock.release()


video_man = threading.Thread(target=video_manage)
audio_man = threading.Thread(target=audio_manage)
video_man.start()
audio_man.start()
print("starting thread0")

