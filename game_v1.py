# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from pynput.mouse import Button, Controller
import socket
import threading
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
import json
import math
import random

import pygame
from pygame import mixer

mouse_x = 0
mouse_y = 0


acc_x = 0
acc_y = 0

del_t = 0
start_time=0
init_mouse = 0
def fun1(port_num) :
	global mouse
	localIP     = "192.168.1.4"
	localPort   = port_num
	bufferSize  = 1024
    # Create a datagram socket
	UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
	# Bind to address and ip
	UDPServerSocket.bind((localIP, localPort))
	print("UDP server up and listening")
	# Listen for incoming datagrams
	prev_time = -1
	global acc_x
	global acc_y
	global theta
	global shoot
	shoot = 0
	while True:
		bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
		curr_time = time.time()
		if prev_time==-1 :
			prev_time=curr_time

		del_t = curr_time - prev_time
		prev_time = curr_time
		clientMsg = bytesAddressPair[0]
		s=json.loads(clientMsg)
		# print(s)
		if s['purpose'] == 'update' :
			acc_x = s['ax']
			acc_y = s['ay']
			theta = s['yaw'] - math.pi/6
		elif s['purpose'] == 'click' :
			if s['click']==-1 :
				shoot = 1
		else :
			print("Screen_size",s['length'],s['breadth'])

		if del_t == 0 :
			kalman_init()
		kalman(del_t)
  
  
def fun2() :
	

	# construct the argument parse and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument("-p", "--prototxt", required=True,
	# 	help="path to Caffe 'deploy' prototxt file")
	# ap.add_argument("-m", "--model", required=True,
	# 	help="path to Caffe pre-trained model")
	# ap.add_argument("-c", "--confidence", type=float, default=0.2,
	# 	help="minimum probability to filter weak detections")
	# args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect, then generate a set of bounding box colors for each class
	COLORS = np.random.uniform(0, 255, size=(3, 3))

	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')

	# initialize the video stream, allow the cammera sensor to warmup,
	# and initialize the FPS counter
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)
	fps = FPS().start()
	count = 0
	x = 0
	y=0
	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		height, width = frame.shape[:2]


		img = frame
		rows = img.shape[0]
		cols = img.shape[1]
		net.setInput(cv2.dnn.blobFromImage(img, size=(300, 300), swapRB=True, crop=False))
		cvOut = net.forward()

		for detection in cvOut[0,0,:,:] :
			idx = detection[1]
			score = float(detection[2])
			
			if score > 0.3 and int(idx) == 77:
				left = detection[3] * cols
				top = detection[4] * rows
				right = detection[5] * cols
				bottom = detection[6] * rows
				label = "{}: {:.2f}%".format("phone",score* 100)
				global mouse_x
				global mouse_y
				
				cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
				mouse_x = int(2500*(left+right)*0.5/width)
				mouse_y = int(1406*(top+bottom)*0.5/height)
				mouse_y = mouse_y - 300
				mouse_x = 2050 -mouse_x
				if mouse_x > 1920 :
					mouse_x = 1920
				elif mouse_x < 0 :
					mouse_x = 0
				if mouse_y > 1080 :
					mouse_y = 1080
				elif mouse_y < 0 :
					mouse_y = 0
				x += mouse_x
				y += mouse_y
				count +=1
				global init_mouse
				init_mouse = 1
				# print("init mouse 1")
				if count==5 :
					count=0
					# print("Camera",x/5,y/5)
					x,y=0,0
		# show the output frame
		
		# update the FPS counter
		fps.update()

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()
 
c_x = 0.01515088909
c_y = 0.02248852815

def kalman_init() :
	while not init_mouse :
		i=1
	x_position = mouse_x*c_x/100
	y_position = mouse_y*c_y/100
	global kf_x
	kf_x = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
	kf_x.x = np.array([x_position, 0.])

	kf_x.P = [[1,0],[0,1]]
	kf_x.Q = [[0.1,0],[0,0.1]]
	kf_x.R = [[1]]
	kf_x.H = np.array([[1.,0.]])

	global kf_y
	kf_y = KalmanFilter(dim_x=2, dim_z=1, dim_u=1)
	kf_y.x = np.array([y_position, 0.])

	kf_y.P = [[1,0],[0,1]]
	kf_y.Q = [[0.1,0],[0,0.1]]
	kf_y.R = [[10]]
	kf_y.H = np.array([[1.,0.]])
	print("Kalman Initialized")
 
def kalman(del_t1) :
	global mouse
	position_x = mouse_x*c_x/100
	position_y = mouse_y*c_y/100
	#update matrices
	global kf_x
	kf_x.F = np.array([[1.,del_t1],
				[0.,1.]])

	kf_x.B = np.array([0.5*(del_t1**2), del_t1])

	#predict new values
	kf_x.predict(acc_x)
	kf_x.update(position_x)

	#update matrices
	global kf_y 
	kf_y.F = np.array([[1.,del_t1],
				[0.,1.]])
	kf_y.B = np.array([0.5*(del_t1**2), del_t1])

	#predict new values
	kf_y.predict(acc_y)
	kf_y.update(position_y)

	filtered_x = kf_x.x[0]
	filtered_y = kf_y.x[0]
	global x_kalman
	global y_kalman
	x_kalman = (filtered_x*100/c_x)*800/1920
	y_kalman = (filtered_y*100/c_y)*800/1080
	# mouse.position = (filtered_x*100/c_x,filtered_y*100/c_y)
	# print("Kalman",kf_x.x,kf_y.x)
	# print("Kalman",filtered_x*100/c_x,filtered_y*100/c_y)
	
 
 
 

if __name__ == "__main__":
	global mouse
	mouse = Controller()
	# creating thread
	# t1 = threading.Thread(target=kalman, args=())
	t2 = threading.Thread(target=fun1, args=(5555,))
	t3 = threading.Thread(target=fun2, args=())
	# t3 = threading.Thread(target=server,args=(8081,))

	t2.start()
	t3.start()
	time.sleep(5)
    
    # t1.start()
	pygame.init()

	# create the screen
	screen = pygame.display.set_mode((800, 800))

	# Background
	background = pygame.image.load('game_bg.jpg') # Put relevant bg pic

	# Sound

	# Caption and Icon
	pygame.display.set_caption("Village saviour")  # Change accordingly
	icon = pygame.image.load('ufo.png')
	pygame.display.set_icon(icon)

	# Player
	playerImg = pygame.image.load('game_player.png')  # put relevant pic for player
	playerX = 550
	playerY = 550

	# Enemy
	enemyImg = []
	enemyX = []
	enemyY = []
	num_of_enemies = 3
	num_of_frames = 3
	R = 220

	for i in range(num_of_enemies):
		enemyImg.append([pygame.image.load('enemy_frame1.png'),pygame.image.load('enemy_frame2.png'),pygame.image.load('enemy_3.png')])
		enemyX.append(random.randint(0, 736))   # modify range
		enemyY.append(random.randint(50, 500))  # modify range

	# Bullet

	# Ready - You can't see the bullet on the screen
	# Fire - The bullet is currently moving

	bulletImg = pygame.image.load('bullet.png')  # Put bullet pic
	bulletX = 0
	bulletY = 480
	bullet_change = 5
	bulletX_change = 10
	bulletY_change = 10
	bullet_state = "ready"

    # Score

	score_value = 0
	font = pygame.font.Font('freesansbold.ttf', 32)

	textX = 10
	testY = 10

    # # Game Over
    # over_font = pygame.font.Font('freesansbold.ttf', 64)


	def show_score(x, y):
		score = font.render("Score : " + str(score_value), True, (0, 0, 0))
		screen.blit(score, (x, y))


    # def game_over_text():
    #     over_text = over_font.render("GAME OVER", True, (255, 255, 255))
    #     screen.blit(over_text, (200, 250))


	def player(x, y, angle):
		rotated_player = pygame.transform.rotate(playerImg, angle*180/math.pi)
		screen.blit(rotated_player, (x, y))


	def enemy(x, y, i, j): # j is for the illusion. Multiple frames
		screen.blit(enemyImg[i][j], (x, y))


	def fire_bullet(x, y):
		global bullet_state
		bullet_state = "fire"
		screen.blit(bulletImg, (x , y )) # replace the ?

	# Print one circle to specify the player's shooting range : Radius R
	def circle(x, y, R):
		BLUE = (135, 206, 235)
		BLACK = (0,0,0)
		pygame.draw.circle(screen, BLACK, (x+40,y+40), R-3, width=2)
		pygame.draw.circle(screen, BLUE, (x+40,y+40), R, width=2)
		pygame.draw.circle(screen, BLACK, (x+40,y+40), R+3, width=2)

	def calc_distance(x1, y1, x2, y2):
		distance = math.sqrt(math.pow(x1 - x2, 2) + (math.pow(y1 - y2, 2)))
		return distance

	def isCollision(enemyX, enemyY, bulletX, bulletY):
		distance = math.sqrt(math.pow(enemyX - bulletX, 2) + (math.pow(enemyY - bulletY, 2)))
		if distance < 50: # Change distance
			return True
		else:
			return False

	j = 0
	# Game Loop
	running = True
	while running:

		# RGB = Red, Green, Blue
		screen.fill((0, 0, 0))
		# Background Image
		screen.blit(background, (0, 0))
		global x_kalman
		global y_kalman
		global theta
		playerX = x_kalman
		playerY = y_kalman
		if playerX <= 100:  # Set limits, modify values accordingly
			playerX = 100
		elif playerX >= 700:
			playerX = 700 # Similarly do for y also
			
		if playerY <= 100:  # Set limits, modify values accordingly
			playerY = 100
		elif playerY >= 700:
			playerY = 700
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				running = False
			# if keystroke is pressed check whether its right or left
		
  
		global shoot
		if shoot :   # Change bullet from space to phone button
			if bullet_state is "ready":
				# bulletSound = mixer.Sound("laser.wav")
				# bulletSound.play()
				# Get the current x cordinate of the spaceship
				bulletY = playerY + 40 # Set bullet y's value based on player's position
				bulletX = playerX + 40# Set bullet x's value based on player's position
				bullet_startingY = bulletY
				bullet_startingX = bulletX
				fire_bullet(bulletX, bulletY)
				shoot = 0
	

        # 5 = 5 + -0.1 -> 5 = 5 - 0.1
        # 5 = 5 + 0.1
		
            
		circle(playerX, playerY, R)  # Displaying circle for player's range 

        # Enemy Movement
		for i in range(num_of_enemies):

            # Collision
			collision = isCollision(enemyX[i], enemyY[i], bulletX, bulletY)
			if collision:
				bullet_state = "ready"
				score_value += 1
				enemyX[i] = random.randint(0, 736)
				enemyY[i] = random.randint(50, 500)

			enemy(enemyX[i], enemyY[i], i, j)
			j += 1
			if j == 3: j=0

		if bullet_state == "fire" :
			if calc_distance(bulletX, bulletY, bullet_startingX, bullet_startingY ) >= R: # R is radius of range
				bullet_state = "ready"
        
		if theta > math.pi/3: theta = math.pi/3
		elif theta < -1*math.pi/3: theta = -1*math.pi/3
		
		bulletY_change , bulletX_change =  bullet_change*math.cos(theta) , bullet_change*math.sin(theta)
		if bullet_state is "fire":
			fire_bullet(bulletX, bulletY)
			bulletY -= bulletY_change
			bulletX -= bulletX_change

		player(playerX, playerY, theta)
		show_score(textX, testY)
		pygame.display.update()


    # t1.join()
	t2.join()
	t3.join()
	# both threads completely executed
	print("Done!")