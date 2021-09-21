# **Face_Mask_detection**
 <p align="center">
   <img src="https://github.com/xoikia/Face_mask_detection/blob/main/logo.jpg" alt="LOGO">
</p>

Due to covid-19 pandemic it has become very important for everyone to wear face mask not just for protection of itself but also for others. This model detects whether an 
individual is wearing a mask or not in real time, thus ensuring your safety and the safety of others .

# **Dataset**

The [dataset](https://github.com/xoikia/Face_mask_detection/tree/main/Face_Mask_Detection/dataset) consists of 1376 images with 690 images containing images of people 
wearing masks and 686 images with people without masks.

# **Builidng the Neural Network**

The workflow of the project is  to first build a neural network and then train on the dataset and save the model then finally implement in real time to detect whether 
an individual is wearing a mask or not

The directory consist of 4 files.

[train](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/train.py) builds a neural network model and then train 
the model in the dataset.

[create_plots](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/create_plots.py) is used to create the model's various evaluation metrics in
the form of graphs, It will create two plots [confusion matrix](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/Confusion%20Matrix.png), 
[accuracy](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/plot.png).

[detect](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/detect.py) is used to detect faces in realtime which will be further fed to our model.

[main](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/main.py) is the final file which will be used to detect mask or nomask in faces in real time.

### **Steps**
      
***Training***

   * We have used the keras MobileNetV2 architecture in my model. This model is already pretrained and thus very effective in feature extraction. On top of this 
     We have added Forward Connection layers consisting of Pooling, Dropout and Dense layers.The Dropout layer is added to avoid overfitting of the model. The 
     final layer of the model is the dense layer with two neurons for classification of mask.
     
     ```
     basemodel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
     top = basemodel.output
     top = AveragePooling2D(pool_size=(7, 7))(top)
     top = Flatten(name="Flatten")(top)
     top = Dense(128, activation="relu")(top)
     top = Dropout(0.5)(top)
     top = Dense(2, activation="softmax")(top)
     ```
     
      
   * After this we trained the model on the dataset and after completion saved the model which saves all the details necessary to reconstitute the model.
     ```
     H = model.fit(data_aug.flow(trainX, trainY, batch_size=BATCHSIZE), steps_per_epoch=len(trainX)//BATCHSIZE,
              validation_data=(testX, testY), epochs=EPOCHS)
     model.save("mask_detector.model", save_format="h5")
     ```
      
   * After training is completed we will also get the Accuracy plot for training and validation data along with the accuracy matrix plot.
     ```
     make_confusion_matrix(testY.argmax(axis=1), pred, group_names=['TN', 'FP', 'FN', 'TP'], categories=lb.classes_)
     create_training_loss_accuracy(model=H, epochs=EPOCHS)
     ```

***Model Evaluations***

<p align="center">
   <img src="https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/plot.png" alt="Accuracy_Loss">
</p>

   The training accuracy was below the validation accuracy at the starting this was because the model was trying to learn, As the training period increases both
   the training  and validation were accuracy were almost same for all of the epochs, Thus we can conclude that our  model was performing well and was not 
   overfitting.
   
   
   
 <p align="center">
   <img src="https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/Confusion%20Matrix.png" alt="Confusion Matrix">
</p>

   The Confusion matrix clearly shows that our model was clearly able to classify  people with mask and without mask correctly.
      
***Implementing in Real-Time***
    
   * We have utilised the DNN Face Detector model along with Opencv to detect faces in real time. It is a Caffe model which is based on the Single Shot-Multibox 
     Detector (SSD) and uses ResNet-10 architecture as its backbone. I haved downloaded the Caffe prototxt file and the  pretrained Caffe model.
     
   * Loading both the face detector model and the mask detector model.
      ```
      faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
      maskNet = load_model("mask_detector.model")
      ```
      
   * The [detect.py](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/detect.py) consists of a detect_and_predict function which accepts the Face              Detector model, the maskdetector model and the frames as parameters. This function detects the face and the coordinates and finally predicts whether the frame has
     mask or not.
     ```
     blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))
     ```
     The dnn.blobFromImage does the preprocessing of the frames which includes setting the blob dimensions and normalization. It creates 4-dimensional blob from image.                Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
     
     To detect facee we pass the blob to the facenet 
     ```
     faceNet.setInput(blob)
     detections = faceNet.forward()
     ```
     Then looping over all the detections the faceNet was able to detect
     ```
     for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            startX, startY = (max(0, startX), max(0, startY))
            endX, endY = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))
      ```
      
      For each detection we extract the confidence score and select only those detections which have confidence score more than 0.5, then proceeded to 
      calculate the co-ordinates of the ROI and then converting back to RGB and resize it to (224,224) so that it matches the input of our maskdetection
      model. Finally appending all the faces along with the locations of the ROI.
      
       Making predictions if at least one face was detected
     ```
     if len(faces) > 0:
         faces = np.array(faces, dtype="float32")
         preds = maskNet.predict(faces, batch_size=32)
     return (locs, preds)
     ```
    
      Finally the function returns the locations and the prediction result of the maskdetection model.
      
      
  * We called this detect_and_predict_mask function inside the while loop in the [main.py](https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/main.py)
    ```
    while True:
      frame = vs.read()
	   frame = imutils.resize(frame, width=1000)

	   locs, preds = detect_and_predict_mask(frame, faceNet, maskNet)
      
      for (box, pred) in zip(locs, preds):
		   startX, startY, endX, endY = box
		   mask, withoutMask = pred
         label = "Mask" if mask > withoutMask else "No Mask"
		   color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
         label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
		   cv2.rectangle(frame, (startX, startY), (endX, endY), color, 3)
		   cv2.rectangle(frame, (startX, startY-35), (endX, endY), color, 3)
		   cv2.putText(frame, label, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)

	   cv2.imshow("Frame", frame)
	   key = cv2.waitKey(1) & 0xFF
	   if key == ord("q"):
		   break
     ```
     
    For every frames the detect_and_predict process it returns the locations and the predictions. Iterating over this values we calculated the coordinates of the 
    bounding boxes and then storing the pred values into *mask*, *withoutMask*. If the model detects an indiviudal with mask then *mask* > *withoutMask* else viceversa,
    based on the above score we create our label. Finally we draw the rectangle boxes over the face and put the text showing the label and the score of the detection.
    
    
# **Model Results**

<p align="center">
   <img src="https://github.com/xoikia/Face_mask_detection/blob/main/Face_Mask_Detection/Video/Demo.gif" alt="Results">
</p>

The model was able to predict whether a person is wearing a mask or not in real time with good accuracy, and if a person keeps his mask in his/her chin the model would detect that they were not wearing any mask.

      
    
    
    
    
    

      
    
    
