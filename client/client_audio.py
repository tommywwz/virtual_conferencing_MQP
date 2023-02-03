import pyaudio
import socket
from socket_client import HOST_IP, PORT
import threading

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

if __name__ == '__main__':
    def send_audio(conn):
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            conn.sendall(data)


    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_addr = (HOST_IP, PORT+1)
    client.connect(socket_addr)
    print("[Client] Got connection")

    send_audio(client)

# send_thread = threading.Thread(target=send_audio, args=(client,))
# send_thread.start()