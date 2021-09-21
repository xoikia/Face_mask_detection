from create_plots import make_confusion_matrix
from create_plots import create_training_loss_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import numpy as np
import os


INIT_LR = 1e-4
EPOCHS = 20
BATCHSIZE = 32

DIRECTORY = r"C:\Mask_Detection\dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)

data_aug = ImageDataGenerator(rotation_range=20,
                              zoom_range=0.15,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.15,
                              horizontal_flip=True,
                              fill_mode='nearest')


basemodel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))


top = basemodel.output
top = AveragePooling2D(pool_size=(7, 7))(top)
top = Flatten(name="Flatten")(top)
top = Dense(128, activation="relu")(top)
top = Dropout(0.5)(top)
top = Dense(2, activation="softmax")(top)

model = Model(inputs=basemodel.input, outputs=top)

for layer in basemodel.layers:
    layer.trainable = False


opt = Adam(learning_rate=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


H = model.fit(data_aug.flow(trainX, trainY, batch_size=BATCHSIZE),
              steps_per_epoch=len(trainX)//BATCHSIZE,
              validation_data=(testX, testY),
              epochs=EPOCHS)

pred = model.predict(testX, batch_size=BATCHSIZE)

pred = np.argmax(pred, axis=1)

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

make_confusion_matrix(testY.argmax(axis=1),
                      pred,
                      group_names=['TN', 'FP', 'FN', 'TP'],
                      categories=lb.classes_)


create_training_loss_accuracy(model=H, epochs=EPOCHS)