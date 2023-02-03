import pyaudio
import socket
from socket_server import HOST_IP, PORT

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


def play_audio(conn):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)
    while True:
        data = conn.recv(CHUNK)
        stream.write(data)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
socket_addr = (HOST_IP, PORT+1)
server.bind(socket_addr)
server.listen(1)
client, addr = server.accept()
play_audio(client)
# play_thread = threading.Thread(target=play_audio, args=(conn,))
# play_thread.start()
