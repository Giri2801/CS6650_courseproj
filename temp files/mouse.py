import pynput
from pynput.mouse import Button,Controller
import time

mouse = Controller()
for i in range(20) :
    print(mouse.position)
    mouse.click(Button.left,1)
    time.sleep(0.5)
    
