from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import  VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf 
IMAGE_SIZE = [224, 224]

train_path = 'E:\newdatasets\train'
valid_path = 'E:\newdatasets\test'

# add preprocessing layer to the front of VGG
vgg =  VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
vgg.summary()
Model: "vgg16"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
=================================================================
Total params: 14,714,688
Trainable params: 14,714,688
Non-trainable params: 0
___________________________
for layer in vgg.layers:
    layer.trainable = False
# useful for getting number of classes
folders = glob('E:/newdatasets/train/*')
folders 
['E:/newdatasets/train\\alexandra',
 'E:/newdatasets/train\\robert',
 'E:/newdatasets/train\\scarlett']

x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()
Model: "model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         (None, 224, 224, 3)       0         
_________________________________________________________________
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 25088)             0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 75267     
=================================================================
Total params: 14,789,955
Trainable params: 75,267
Non-trainable params: 14,714,688
_________________________________________________________________
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'E:\newdatasets\train',target_size = (224, 224),batch_size = 32,class_mode = "categorical")
test_set = test_datagen.flow_from_directory(r'E:\newdatasets\test', target_size = (224, 224),batch_size = 32,class_mode = "categorical")

Found 539 images belonging to 3 classes.
Found 120 images belonging to 3 classes.
# fit the model
r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=15,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set))
  Epoch 1/15
17/17 [==============================] - 136s 8s/step - loss: 1.4203 - accuracy: 0.5232 - val_loss: 0.6091 - val_accuracy: 0.6833
Epoch 2/15
17/17 [==============================] - 141s 8s/step - loss: 0.4480 - accuracy: 0.8089 - val_loss: 0.1913 - val_accuracy: 0.8833
Epoch 3/15
17/17 [==============================] - 134s 8s/step - loss: 0.3141 - accuracy: 0.8664 - val_loss: 0.4260 - val_accuracy: 0.8583
Epoch 4/15
17/17 [==============================] - 131s 8s/step - loss: 0.2350 - accuracy: 0.9147 - val_loss: 0.2378 - val_accuracy: 0.8917
Epoch 5/15
17/17 [==============================] - 131s 8s/step - loss: 0.1635 - accuracy: 0.9555 - val_loss: 0.3034 - val_accuracy: 0.9250
Epoch 6/15
17/17 [==============================] - 131s 8s/step - loss: 0.1833 - accuracy: 0.9295 - val_loss: 0.2122 - val_accuracy: 0.9083
Epoch 7/15
17/17 [==============================] - 137s 8s/step - loss: 0.1527 - accuracy: 0.9406 - val_loss: 0.2271 - val_accuracy: 0.9167
Epoch 8/15
17/17 [==============================] - 133s 8s/step - loss: 0.0998 - accuracy: 0.9833 - val_loss: 0.1253 - val_accuracy: 0.9167
Epoch 9/15
17/17 [==============================] - 134s 8s/step - loss: 0.0893 - accuracy: 0.9759 - val_loss: 0.1116 - val_accuracy: 0.9250
Epoch 10/15
17/17 [==============================] - 131s 8s/step - loss: 0.0969 - accuracy: 0.9722 - val_loss: 0.2324 - val_accuracy: 0.9083
Epoch 11/15
17/17 [==============================] - 135s 8s/step - loss: 0.0949 - accuracy: 0.9777 - val_loss: 0.1888 - val_accuracy: 0.9500
Epoch 12/15
17/17 [==============================] - 134s 8s/step - loss: 0.0735 - accuracy: 0.9870 - val_loss: 0.1547 - val_accuracy: 0.9250
Epoch 13/15
17/17 [==============================] - 135s 8s/step - loss: 0.0591 - accuracy: 0.9963 - val_loss: 0.1710 - val_accuracy: 0.9417
Epoch 14/15
17/17 [==============================] - 130s 8s/step - loss: 0.0639 - accuracy: 0.9833 - val_loss: 0.1952 - val_accuracy: 0.9333
Epoch 15/15
17/17 [==============================] - 126s 7s/step - loss: 0.0557 - accuracy: 0.9907 - val_loss: 0.1773 - val_accuracy: 0.9417

model.save('model-015.model')
# Importing the libraries
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from keras.applications.vgg19 import preprocess_input
import base64
from io import BytesIO
import json
import random
import cv2
from keras.models import load_model
import numpy as np

from keras.preprocessing import image
model = load_model('model-015.model')

# Loading the cascades
face_cascade = cv2.CascadeClassifier('E:\OpenCV-Python-Series-master\src\cascades\data\haarcascade_frontalface_default.xml')

def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    
    if faces is ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face

# Doing some Face Recognition with the webcam
video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    #canvas = detect(gray, frame)
    #image, face =face_detector(frame)
    
    face=face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        im = Image.fromarray(face, 'RGB')
           #Resizing into 128x128 because we trained the model with this image size.
        img_array = np.array(im)
                    #Our keras model used a 4D tensor, (images x height x width x channel)
                    #So changing dimension 128x128x3 into 1x128x128x3 
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)
                     
        name="None matching"
        
        if(pred[0][1]>0.5):
            name='alexa'
       
       
        cv2.putText(frame,name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(frame,"No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()

       

