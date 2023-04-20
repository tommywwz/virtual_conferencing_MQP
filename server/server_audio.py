import pyaudio
import socket
import threading
import select
from Utils.Params import HOST_IP, PORT

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
        inputs = [client]
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)
        while True:
            readable, writable, exceptional = select.select(inputs, [], inputs)
            if exceptional:
                # The client socket has been closed abruptly
                client.close()
                inputs.remove(client)
                print(str(self.addr) + ": abruptly exit")
                break

            data = self.client_sock.recv(CHUNK)
            if not data:
                # The client socket has been closed gracefully
                client.close()
                inputs.remove(client)
                print(str(self.addr) + ": gracefully exit")
                break

            stream.write(data)


if __name__ == '__main__':
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_addr = (HOST_IP, PORT + 1)
    server.bind(socket_addr)
    server.listen(5)

    while True:
        client, client_addr = server.accept()
        if client:
            print("Got audio connection from: " + str(client_addr))
            clientThread = ClientHandler(client_addr, client)
            clientThread.start()

    server.close()

