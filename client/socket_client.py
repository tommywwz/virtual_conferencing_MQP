# TCP reference: https://pyshine.com/Socket-programming-and-openc/
# UDP_reference: https://pyshine.com/Send-video-over-UDP-socket-in-Python/
import cv2
import socket
import base64
import numpy as np

BUFF_SIZE = 65536
host_ip = '192.168.1.143'  # host ip
port = 9999

# create socket
client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
client_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)

message = b'Hello'
client_sock.sendto(message, (host_ip, port))

while True:
    packet, _ = client_sock.recvfrom(BUFF_SIZE)
    data = base64.b64decode(packet, ' /')
    npdata = np.fromstring(data, dtype=np.uint8)
    frame = cv2.imdecode(npdata, 1)
    cv2.imshow("RECEIVING VIDEO", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        client_sock.close()
        break
