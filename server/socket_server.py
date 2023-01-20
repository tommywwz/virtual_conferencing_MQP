# TCP_reference: https://pyshine.com/Socket-programming-and-openc/
# UDP_reference: https://pyshine.com/Send-video-over-UDP-socket-in-Python/
import cv2
import socket
import base64
import imutils

PORT = 9999
BUFF_SIZE = 65536

server_sock = socket.socket(socket.AF_INET,
                            socket.SOCK_DGRAM)
server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
host_name = socket.gethostname()
host_ips = socket.gethostbyname_ex(host_name)
host_ip = "192.168.1.3"
print('host ip: ' + host_ip)

socket_addr = (host_ip, PORT)

# socket bind
server_sock.bind(socket_addr)
print('Listening at: ', socket_addr)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
vid.set(3, 640)  # width
vid.set(4, 360)  # height
while True:
    msg, client_addr = server_sock.recvfrom(BUFF_SIZE)
    print('GOT CONNECTION FROM: ', client_addr, ' says: ', msg.decode('utf-8'))
    while vid.isOpened():
        success, frame = vid.read()
        if success:
            print("frame length " + str(frame.size))
            encoded, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print("\tbuffer length "+str(len(buffer)))
            message = base64.b64encode(buffer)
            print("\t\tmessage length "+str(len(message)))
            server_sock.sendto(message, client_addr)

            cv2.imshow('Transitting Video', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                server_sock.close()
                break


