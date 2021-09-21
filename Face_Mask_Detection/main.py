from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from detect import detect_and_predict_mask
import imutils
import cv2

# loading our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading the face mask detector model from disk
maskNet = load_model("mask_detector.model")

# initializing the video stream
vs = VideoStream(src=0).start()

# iterating over the frames from the video stream
while True:
	'''
	Grabbing the frame from the threaded video stream and resize it to have a maximum width 
	of 1000 pixels
    '''

	frame = vs.read()
	frame = imutils.resize(frame, width=1000)

	'''
	detect faces in the frame and determine if they are wearing a
	face mask or not
	'''

	locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)

	'''
	looping over the detected face locations and their corresponding
	locations
	'''
	for (box, pred) in zip(locs, preds):

		'''
		unpack the bounding box and predictions
		(startX, startY) represents the top left corner of rectangle
		(endX, endY) represents the bottom right corner of rectangle
		'''

		startX, startY, endX, endY = box
		mask, withoutMask = pred

		# determine the class label and color we'll use to draw the bounding box and text
		label = "Mask" if mask > withoutMask else "No Mask"
		print(label)
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# including  the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		print(label)

		# displaying the label and bounding box rectangle on the output frame
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
		cv2.rectangle(frame, (startX, startY-35), (endX, endY), color, 3)
		cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)


	# showing the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()