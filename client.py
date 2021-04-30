
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
    count = 1000
    prev_time = time.time()
    acc_x=[]
    acc_y = []
    while count:
        count-=1
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        curr_time = time.time()
        delta = curr_time-prev_time
        prev_time = curr_time
        # print(delta)
        clientMsg = bytesAddressPair[0]
        # print(clientMsg)
        s=json.loads(clientMsg)
        # print(str(s))
        if s['purpose'] == 'update' :
            accx = s['ax']
            accy = s['ay']
            acc_x.append(accx)
            acc_y.append(accy)
        elif s['purpose'] == '' :
            click = True
            
    plt.plot(np.linspace(0,len(acc_x),len(acc_x)),acc_x)
    plt.plot(np.linspace(0,len(acc_x),len(acc_x)),acc_y)
    plt.legend()
    plt.show()

  
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