from google.colab import drive

drive.mount("/content/drive")

import os
from os import listdir

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from IPython.display import Image, display
from keras import backend as k
from keras import losses 
from keras import optimizers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy


def convert_image_to_array(image_dir):
    
    try:
        
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image,tuple((96,96)))
            return img_to_array(image)
        else:
            return img_to_array(image)
        
    except Exception as e:
        
        print(f"Error : {e}")
        return None


image_list, label_list = [], []
images_directory = os.getcwd()+"/drive/MyDrive/Hackathons/Reva_Hackathon/PlantVillage/PlantVillage"
try:
    print("[INFO] Loading Images....")
    root_dir = listdir(images_directory)
    
    for disease_folder in root_dir:
        print("[INFO] Processing "+disease_folder+"....")
        plant_disease_folder_list = listdir(f"{images_directory}/{disease_folder}")
        
        for image in plant_disease_folder_list[:400]:
            image_directory = f"{images_directory}/{disease_folder}/{image}"
            
            if image_directory.endswith(".jpg") or image_directory.endswith(".JPG"):
                
                image_list.append(convert_image_to_array(image_directory))
                label_list.append(disease_folder)
                
    print("[INFO] Image Loading Completed....")
            
         
        
except Exception as e:
    print(f"Error : {e}")
np_image_list = np.array(image_list, dtype=np.float16)/225.0

print("[INFO] Splitting training and testing datasets....")
x_train, x_test, y_train, y_test = train_test_split(np_image_list, label_list, test_size=0.25, random_state = 42)
print("[INFO] Done splitting datasets....")
aug = ImageDataGenerator(
    rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True,
    fill_mode='nearest')

height = 96
width = 96
depth = 3
model = Sequential() 
inputShape = (height, width, depth)
chanDim = -1
if k.image_data_format() == 'channels_first':
    inputShape = (depth, height, width)
    chanDim = 1


model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3,3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(64,(3,3), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (2,2), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(Conv2D(128, (2,2), padding="same", input_shape=inputShape))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation("relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Activation("softmax"))
model.summary()
opt = Adam(lr=1e-3, decay=1e-3 / 5)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] Training Network....")
history = model.fit(
    aug.flow(x_train, y_train, batch_size=32),
    validation_data=(x_test, y_test),
    steps_per_epoch=len(x_train) // 32,
    epochs=5, verbose=1)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accurarcy')
plt.plot(epochs, val_acc, 'r', label='Validation accurarcy')
plt.title('Training and Validation accurarcy')
plt.legend()

plt.figure()
#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy : {scores[1]*100}")

print("[INFO] Saving model....")
model.save(model,open('cnn_model.pkl', 'wb'))
print("[INFO] Done Saving model....")