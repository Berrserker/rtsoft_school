from flask import Flask, url_for, render_template, Response

import cv2       # OpenCV
import torch     # PyTorch
import matplotlib.pyplot as plt  # для отрисовки изображений
import numpy as np
set_up = False;

from multiprocessing import Process, Pipe

app = Flask(__name__)
parent_conn, child_conn = Pipe()


def worker(conn):

	cap = cv2.VideoCapture(0)

	bool = True;

	try:
		bool = conn.recv()
	except:
		pass

	while(bool):

		ret, img = cap.read()
		blob = cv2.dnn.blobFromImage(img,scalefactor=1/255.0,size=(416,416),swapRB=True)
		net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')
		net.setInput(blob)
		result = net.forward(net.getUnconnectedOutLayersNames())
		boxes = []
		confidences = []
		classIDs = []
		W = img.shape[1]
		H = img.shape[0]
		# loop over each of the layer outputs
		for output in result:
			# loop over each of the detections
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > 0.4:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")
					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
		with open('coco.names', 'r') as labels_file:
			LABELS = labels_file.read().split()
		image = img.copy()
		# ensure at least one detection exists
		if len(idxs) > 0:
			counter = 0
			# loop over the indexes we are keeping
			for i in idxs.flatten():
				if classIDs[i] == 0:
					counter +=1
					# extract the bounding box coordinates
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])
					# draw a bounding box rectangle and label on the image
					color = [0, 255, 0]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
					# print(classIDs[i])
					cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
					cv2.putText(image, str(counter), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
		cv2.imshow("Depth Image", image)
		cv2.waitKey(1);

	cap.release()
	cv2.destroyAllWindows()

app = Flask(__name__)
parent_conn, child_conn = Pipe()

p = ""

@app.route('/', methods=['POST'])
def root():
	return jsonify({'msg' : 'Try POSTing to the /web endpoint to camera work /stop to stop'})

@app.route('/web', methods=['post'])
def start():
	parent_conn, child_conn = Pipe()
	global p
	p = Process(target=worker, args=(child_conn,), daemon=True)
	set_up = True
	p.start()
	parent_conn.send(set_up)
	return "Send startsignal"
	# worker('Tr')

@app.route('/stop', methods=['post'])
def stop():
	set_up = False
	parent_conn.send(set_up)
	p.terminate()
	return "Send stopsignal"

if __name__ == '__main__':
    app.run()
