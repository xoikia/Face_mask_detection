# **Face_Mask_detection**

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

    * I have used the keras MobileNetV2 architecture in my model. This model is already pretrained and thus very effective 
      in feature extraction. On top of this I have added Forward Connection layers consisting of Pooling, Dropout and Dens-
      -e layers.The Dropout layer is added to avoid overfitting of the model. The final layer of the model is the dense la-
      -yer with two neurons for classification of mask.
      
    * After this I trained the model on the dataset and after completion saved the model which saves all the details nece-
      -ssary to reconstitute the model.
      
    * After training is completed we will also get the Accuracy plot for training and validation data along with the accura-
      -cy matrix plot.
    
    
