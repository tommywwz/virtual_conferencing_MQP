# reference: https://pyshine.com/Socket-programming-and-openc/
# require soundfile, sounddevice
import cv2
import pickle
import socket
import struct
import threading
import select
from wheels.Frame import Frame

PORT = 9999
buff_4K = 4 * 1024
HOST_IP = '192.168.1.3'

# Define the number of channels, sample rate, and sample width
channels = 2
sample_rate = 44100
sample_width = 2

if __name__ == '__main__':

    class VideoClientThread(threading.Thread):
        def __init__(self, client_socket, client_addr):
            threading.Thread.__init__(self)
            self.client_socket = client_socket
            self.client_addr = client_addr

        def run(self):
            video_client(self.client_socket, self.client_addr)


    def video_client(client_socket, client_addr):
        inputs = [client_socket]
        data = b""
        payload_size = struct.calcsize("Q")
        windowName = str(client_addr)
        cv2.namedWindow(windowName)
        while True:
            readable, writable, exceptional = select.select(inputs, [], inputs)
            if exceptional:
                # The client socket has been closed abruptly
                client_socket.close()
                inputs.remove(client_socket)
                print(str(client_addr) + ": abruptly exit")
                break

            while len(data) < payload_size:
                packet = client_socket.recv(buff_4K)  # 4K
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]  # extracting the packet size information

            if not packed_msg_size:  # check if client has lost connection
                client_socket.close()
                inputs.remove(client_socket)
                cv2.destroyWindow(windowName)
                print("Client:", client_addr, " Exited")
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
                newClientVidThread = VideoClientThread(client_socket, client_addr)
                newClientVidThread.start()
                print("starting thread for client:", client_addr)


    video_man = threading.Thread(target=video_manage)

    video_man.start()
    print("starting thread0")

