import threading, socket, select, struct, pickle
from Utils import Params
import time


class VideoServer:
    def __init__(self, host_ip, port, CamMan):
        self.server_socket = None
        self.host_ip = host_ip
        self.port = port
        self.CamMan = CamMan
        self.exit_event = threading.Event()

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket_addr = (self.host_ip, self.port)
        self.server_socket.bind(server_socket_addr)
        print('host ip: ' + self.host_ip)
        self.server_socket.setblocking(False)  # set to non-blocking accept
        self.server_socket.listen(5)

        while not self.exit_event.is_set():
            try:
                client_socket, address = self.server_socket.accept()
                print(f"Accepted connection from {address}")
                client_socket.setblocking(True)  # set blocking back on
                client_thread = threading.Thread(target=self.clientThread, args=(client_socket, address, ))
                client_thread.start()
            except BlockingIOError:
                time.sleep(0.5)
                pass
        print(Params.OKGREEN + "Closing Client Handler" + Params.ENDC)
        self.server_socket.close()
        print(Params.OKGREEN + "Client Handler Closed" + Params.ENDC)

    def clientThread(self, client_socket, client_addr):
        inputs = [client_socket]
        clientAddr_camID, clt_port = client_addr
        data = b""
        payload_size = struct.calcsize("Q")
        if not self.exit_event.is_set():
            self.CamMan.init_cam(clientAddr_camID)  # initialize the FIFO queue for current camera feed

        while not self.exit_event.is_set():
            readable, writable, exceptional = select.select(inputs, [], inputs)
            if exceptional:
                # The client socket has been closed abruptly
                # client_socket.close()
                inputs.remove(client_socket)
                print(Params.WARNING + str(client_addr) + ": abruptly exit" + Params.ENDC)
                break

            while len(data) < payload_size:
                packet = client_socket.recv(Params.buff_4K)  # 4K
                if not packet:
                    break
                data += packet
            packed_msg_size = data[:payload_size]  # extracting the packet size information

            if not packed_msg_size:  # check if client has lost connection
                # client_socket.close()
                inputs.remove(client_socket)
                print(Params.OKGREEN + "Client:", client_addr, " Exited" + Params.ENDC)
                break

            data = data[payload_size:]  # extract the img data from the rest of the packet
            msg_size = struct.unpack("Q", packed_msg_size)[0]

            while len(data) < msg_size:
                # keep loading the data until the entire data received
                data += client_socket.recv(Params.buff_4K)
            frame_data = data[:msg_size]
            data = data[msg_size:]
            frameClass = pickle.loads(frame_data)
            self.CamMan.put_frame(clientAddr_camID, frameClass)

        client_socket.close()
        self.CamMan.delete_cam(clientAddr_camID)

    # def handle_client(self, client_socket):
    #     while not self.exit_event.is_set():
    #         data = client_socket.recv(1024)
    #         if not data:
    #             break
    #         print(f"Received data: {data}")
    # 
    #     client_socket.close()

    def stop(self):
        self.exit_event.set()

