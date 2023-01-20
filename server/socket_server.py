# reference: https://pyshine.com/Socket-programming-and-openc/
import cv2
import pickle
import socket
import struct

PORT = 9999

server_sock = socket.socket(socket.AF_INET,
                            socket.SOCK_STREAM)
host_name = socket.gethostname()
host_ips = socket.gethostbyname_ex(host_name)
host_ip = host_ips[2][1]
print('host ip: ' + host_ip)

socket_addr = (host_ip, PORT)

# socket bind
server_sock.bind(socket_addr)

# socket listen
server_sock.listen(5)
print('Listening at: ', socket_addr)


while True:
    client_socket, client_addr = server_sock.accept()
    print('GOT CONNECTION FROM: ', client_addr)
    if client_socket:
        vid = cv2.VideoCapture(3, cv2.CAP_DSHOW)
        while vid.isOpened():
            success, frame = vid.read()
            if success:
                pickled_frame = pickle.dumps(frame)

                # data length followed by serialized frame object
                msg = struct.pack("Q", len(pickled_frame))+pickled_frame
                client_socket.sendall(msg)

                cv2.imshow('Transitting Video', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    client_socket.close()

