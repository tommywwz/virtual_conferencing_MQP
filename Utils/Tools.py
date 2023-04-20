import socket

def get_host_ip():
    # create a datagram socket (single UDP request and response, then close)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # connect to an address on the internet, that's likely to always be up
    # (the Google primary DNS is a good bet)
    sock.connect(("8.8.8.8", 80))
    # after connecting, the socket will have the IP in its address
    host_ip = sock.getsockname()[0]
    print("Your Computer IP Address is: " + host_ip)
    # done
    sock.close()
    return host_ip