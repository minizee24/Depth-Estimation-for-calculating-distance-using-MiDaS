import numpy as np
import argparse
import time
import cv2
import os
from MidasDepthEstimation.midasDepthEstimator import midasDepthEstimator
import math

depthEstimator = midasDepthEstimator()

labelsPath = os.path.sep.join(["coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join( ["yolov3.weights"])
configPath = os.path.sep.join(["yolov3.cfg"])
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

image = cv2.imread('image.jpg')
(H, W) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layerOutputs = net.forward(ln)
end = time.time()

print("[INFO] YOLO took {:.6f} seconds".format(end - start))
from google.colab.patches import cv2_imshow

boxes = []
confidences = []
classIDs = []

for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence >0.5:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        mid_x = int((boxes[i][1] + boxes[i][3]) / 2)
        mid_y = int((boxes[i][0] + boxes[i][2]) / 2)
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        colorDepth = depthEstimator.estimateDepth(image)
        cv2.rectangle(colorDepth, (x, y), (x+w, y+h), color, 4)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(colorDepth, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        p1=int(x+w/2)
        p2=int(y+h/2)     
        c = colorDepth[p2,p1]
        d= round((1+(1/(math.sqrt(c[0]**2 + c[1]**2 + c[2]**2)))*100), 2)
        # d = abs(1 - ( math.sqrt(c[0]**2 + c[1]**2 + c[2]**2 ))/100)
        print('DISTANCE = ' + str(d))
        # cv2.circle(image, (p1,p2), 4, (0, 0, 255))

        cv2.putText(image, f" Distance = {d} m", (x-20, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
        cv2.putText(colorDepth, f" Distance = {d} m", (x-40, y ), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        combinedImg = np.hstack((image, colorDepth))
        cv2_imshow(combinedImg)
        if cv2.waitKey(1) == ord('q'):
            break

