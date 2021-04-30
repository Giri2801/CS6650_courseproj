
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
import json
    
def server(port_num) :
    localIP     = "192.168.1.4"
    localPort   = port_num
    bufferSize  = 1024
    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # Bind to address and ip
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
    # Listen for incoming datagrams
    count = 100
    prev_time = time.time()
    while count:
        count-=1
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        curr_time = time.time()
        delta = curr_time-prev_time
        print(delta)
        clientMsg = bytesAddressPair[0]
        print(clientMsg)
        s=json.loads(clientMsg)
        if s['purpose'] == 'update' :
            accx = s['linAccel'][0]
            accy = s['linAccel'][1]
        elif s['purpose'] == '' :
            click = True

  
if __name__ == "__main__":
    # creating thread
    # t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=server, args=(5555,))
    # t3 = threading.Thread(target=server,args=(8081,))
    
    
    # t1.start()
    t2.start()
    # t3.start()

    # t1.join()
    t2.join()
    # t3.join()
    # both threads completely executed
    print("Done!")