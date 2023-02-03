import pyaudio
import socket
import threading
from socket_server import HOST_IP, PORT

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


class ClientHandler(threading.Thread):
    def __init__(self, addr, client_sock):
        threading.Thread.__init__(self)
        self.addr = addr
        self.client_sock = client_sock

    def run(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)
        while True:
            data = self.client_sock.recv(CHUNK)
            stream.write(data)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_addr = (HOST_IP, PORT + 1)
server.bind(socket_addr)
server.listen(5)

while True:
    client, client_addr = server.accept()
    if client:
        clientThread = ClientHandler(client_addr, client)
        clientThread.start()

server.close()

