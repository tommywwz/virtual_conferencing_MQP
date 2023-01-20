# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct

buff_4K = 4*1024

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# create socket
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
host_ip = '192.168.1.3'  # paste your server ip address here
port = 9999
client_socket.connect((host_ip, port))  # a tuple
data = b""
payload_size = struct.calcsize("Q")


while vid.isOpened():
    success, frame = vid.read()
    if success:
        pickled_frame = pickle.dumps(frame)

        # data length followed by serialized frame object
        msg = struct.pack("Q", len(pickled_frame)) + pickled_frame
        client_socket.sendall(msg)

        cv2.imshow('Transitting Video', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            client_socket.close()


# while True:
#     while len(data) < payload_size:
#         packet = client_socket.recv(buff_4K)  # 4K
#         if not packet: break
#         data += packet
#     packed_msg_size = data[:payload_size]
#     data = data[payload_size:]
#     msg_size = struct.unpack("Q", packed_msg_size)[0]
#
#     while len(data) < msg_size:
#         data += client_socket.recv(buff_4K)
#     frame_data = data[:msg_size]
#     data = data[msg_size:]
#     frame = pickle.loads(frame_data)
#     cv2.imshow("RECEIVING VIDEO", frame)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
# client_socket.close()




