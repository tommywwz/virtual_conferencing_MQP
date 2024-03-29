import pyaudio
import socket
import Utils.Params as Params

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


class ClientAudio:
    def __init__(self, HOST_IP, PORT):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket_addr = (HOST_IP, PORT + 1)
        self.client.connect(self.socket_addr)
        print("[Client] Got connection")
        self.send_audio()

    def send_audio(self):
        conn = self.client
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        while True:
            data = stream.read(CHUNK, exception_on_overflow=False)
            conn.sendall(data)


if __name__ == '__main__':
    client_audio = ClientAudio(Params.HOST_IP, Params.PORT)

# send_thread = threading.Thread(target=send_audio, args=(client,))
# send_thread.start()