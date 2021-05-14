# import the necessary packages, Controller
import socket
import threading
import json
import pynput
import math

# from autocorrect import spess

#### Put Autocorrect into the keyboard business ####
# https://github.com/phatpiglet/autocorrect


mButton = pynput.mouse.Button

mouse_x = 0
mouse_y = 0

acc_x = 0
acc_y = 0

del_t = 0
start_time = 0
init_mouse = 0

s = 0

fx = list()
fy = list()


def updateFun(s):
    global mouse
    yaw = s['yaw'] * 180 / math.pi
    pitch = s['pitch'] * 180 / math.pi

    posx = 1920 / 2 - yaw * 32
    posy = 1080 / 2 - pitch * 22 + 1080 / 6
    mouse.position = (posx, posy)


def clickFun(s):
    global mouse
    if s['click'] == -1:
        mouse.click(mButton.left, 1)
    elif s['click'] == -2:
        mouse.click(mButton.left, 2)
    elif s['click'] == 2:
        mouse.click(mButton.right, 2)
    else:
        mouse.click(mButton.right, 1)


def scrollFun(s):
    global mouse
    j = s['scroll']
    mouse.scroll(0, -int(j / 50.0))


def pageFun(s):
    global keyboard
    if s['page'] == 1:
        keyboard.press(pynput.keyboard.Key.page_up)
        keyboard.release(pynput.keyboard.Key.page_up)
    elif s['page'] == -1:
        keyboard.press(pynput.keyboard.Key.page_down)
        keyboard.release(pynput.keyboard.Key.page_down)


backcount = 0


def keyboardFun(s):
    global keyboard, backcount
    keyboard.type(s['key'])
    if s['backspace'] == 1:
        if backcount == 0:
            backcount = backcount + 1
            keyboard.press(pynput.keyboard.Key.backspace)
            keyboard.release(pynput.keyboard.Key.backspace)
        else :
            backcount = 0



def nothingFun(s):
    pass


def fun1(port_num):
    global s
    localIP = socket.gethostbyname(socket.gethostname())
    print("[IP ADDRESS] Set IP to: " + localIP)
    localPort = port_num
    bufferSize = 1024
    # Create a datagram socket
    UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
    # Bind to address and ip
    UDPServerSocket.bind((localIP, localPort))
    print("UDP server up and listening")
    # Listen for incoming datagrams

    while True:
        bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
        clientMsg = bytesAddressPair[0]
        s = json.loads(clientMsg)

        switcher = {
            'update': updateFun,
            'click': clickFun,
            'scroll': scrollFun,
            'keyboard': keyboardFun,
            'page': pageFun,
            'screen': nothingFun
        }

        func = switcher.get(s['purpose'], lambda: nothingFun)
        func(s)


if __name__ == "__main__":
    global mouse
    mouse = pynput.mouse.Controller()
    keyboard = pynput.keyboard.Controller()
    # creating thread
    t2 = threading.Thread(target=fun1, args=(5555,))

    t2.start()
    t2.join()
    print("Done!")
