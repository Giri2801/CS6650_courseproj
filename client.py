
import socket
import threading
import numpy as np
import matplotlib.pyplot as plt
import time
    
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
    accx = []
    accy = []
    accz = []
    gyrox = []
    gyroy = []
    gyroz = []
    count = 500
    while(True):
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        message = bytesAddressPair[0]
        clientMsg = "Message from Client:{}".format(message)  
        data =  clientMsg.split(',')
        j=0
        while j < len(data):
            temp = list([i for i in (data[j]) if (i.isdigit() or i=='.')])
            j+=1
            data[j-1] = ''.join(temp)
        # print(clientMsg)
        # print(data)
        if (len(data)<17) :
            continue
        count -= 1
        accx.append(float(data[14]))
        accy.append(float(data[15]))
        accz.append(float(data[16]))
        gyrox.append(float(data[6]))
        gyroy.append(float(data[7]))
        gyroz.append(float(data[8]))
        if count < 0 :
            break
        
    plt.figure(figsize=(20,10))
    plt.subplot(3,2,1)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),gyrox)
    plt.ylabel("gyrox")
    plt.xlabel("time")
    
    plt.subplot(3,2,2)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),accx)
    plt.ylabel("accx")
    plt.xlabel("time")
    
    plt.subplot(3,2,3)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),gyroy)
    plt.ylabel("gyroy")
    plt.xlabel("time")

    plt.subplot(3,2,4)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),accy)
    plt.ylabel("accy")
    plt.xlabel("time")
    
    plt.subplot(3,2,5)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),gyroz)
    plt.ylabel("gyroz")
    plt.xlabel("time")
    
    plt.subplot(3,2,6)
    plt.plot(np.linspace(0,len(gyrox),num=len(gyrox)),accz)
    plt.ylabel("accz")
    plt.xlabel("time")
    plt.show()
    

  
if __name__ == "__main__":
    # creating thread
    # t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=server, args=(8080,))
    t3 = threading.Thread(target=server,args=(8081,))
    
    
    # t1.start()
    t2.start()
    t3.start()

    # t1.join()
    t2.join()
    t3.join()
    # both threads completely executed
    print("Done!")