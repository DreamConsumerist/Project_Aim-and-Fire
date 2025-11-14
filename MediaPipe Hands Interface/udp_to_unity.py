import socket
import random

UDP_IP = "127.0.0.1"
UDP_PORT = 5555

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

message = str(random.random())
encoded_message = message.encode('utf-8')
sock.sendto(encoded_message, (UDP_IP, UDP_PORT))
sock.close()
print(f"UDP message sent to {UDP_IP}:{UDP_PORT}")
print(f"Message: {message}")
