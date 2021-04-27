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





def fun1() :
	port = 8081
	s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	s.bind(("192.168.1.4", port))
	j = 20
	while j:
		j -= 1
		data,  = s.recvfrom(1024)
		print(data)
  
  
def fun2() :
	mouse = Controller()

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

	# loop over the frames from the video stream
	loop_c = 0
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vs.read()
		frame = imutils.resize(frame, width=400)
		height, width = frame.shape[:2]

		# grab the frame dimensions and convert it to a blob
		# (h, w) = frame.shape[:2]
		# blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		# 	0.007843, (300, 300), 127.5)

		# # pass the blob through the network and obtain the detections and
		# # predictions
		# net.setInput(blob)
		# detections = net.forward()

		# # loop over the detections
		# for i in np.arange(0, detections.shape[2]):
		# 	# extract the confidence (i.e., probability) associated with
		# 	# the prediction
		# 	confidence = detections[0, 0, i, 2]

		# 	# filter out weak detections by ensuring the `confidence` is
		# 	# greater than the minimum confidence
		# 	if confidence > 0.1 :
		# 		# extract the index of the class label from the
		# 		# `detections`, then compute the (x, y)-coordinates of
		# 		# the bounding box for the object
		# 		idx = int(detections[0, 0, i, 1])
		# 		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		# 		(startX, startY, endX, endY) = box.astype("int")

		# 		# draw the prediction on the frame
		# 		label = "{}: {:.2f}%".format(str(idx),
		# 			confidence * 100)
		# 		cv2.rectangle(frame, (startX, startY), (endX, endY),
		# 			COLORS[0], 2)
		# 		y = startY - 15 if startY - 15 > 15 else startY + 15
		# 		cv2.putText(frame, label, (startX, y),
		# 			cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
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
				mouse.position = (mouse_x, mouse_y)
				y = int(top) - 15
				if top < 30 :
					y = int(top) + 15
				cv2.putText(frame, label, (int(left), y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[0], 2)
		# show the output frame

		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break
		
		# update the FPS counter
		fps.update()
		loop_c += 1
		if loop_c > 100 :
			loop_c = 0
			mouse.click(Button.left,1)

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vs.stop()

if __name__ == "__main__":
    # creating thread
    # t1 = threading.Thread(target=print_square, args=(10,))
    t2 = threading.Thread(target=fun1, args=())
    t3 = threading.Thread(target=fun2, args=())
    # t3 = threading.Thread(target=server,args=(8081,))
    
    
    # t1.start()
    t2.start()
    t3.start()

    # t1.join()
    t2.join()
    t3.join()
    # both threads completely executed
    print("Done!")
