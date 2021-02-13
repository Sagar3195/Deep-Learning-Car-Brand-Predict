# -*- coding: utf-8 -*-
"""
### Car Brand Classification Using Deep Learning
"""
###importing required libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

from glob import glob

#resize all images to this shape
IMAGE_SIZE = [224, 224]

#loading dataset
train_path = "/content/drive/My Drive/CarBrand/Datasets/Train"
valid_path = "/content/drive/My Drive/CarBrand/Datasets/Test"

# Import the Resnet50 library as shown below and add preprocessing layer to the front of 
# Here we will be using imagenet weights
#IMAGE_SIZE + [3] = RGB

#now initialize resent50
resnet = ResNet50(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)

resnet.summary()

#now we don't want to train existing weights
for layer in resnet.layers:
    layer.trainable = False

#useful for getting number of output classes
folders = glob("/content/drive/My Drive/CarBrand/Datasets/Train/*")

len(folders) #here we can see that 3 output classes.

folders

x = Flatten()(resnet.output)

#output layer
prediction = Dense(len(folders), activation = 'softmax')(x)



#create model object 
model = Model(inputs = resnet.input, outputs = prediction)

#now we see summary
#here we can see that output layer is added.
model.summary()



#Now we compile the model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Now we use the Image Data Generator to import the images from the dataset
#here we doing data augmentation on train dataset
#image pixel size range from 0 to 255. 
#so here we rescalling size of image.
#Bcoz we image size will be different that why we do rescale image.s
train_datagen = ImageDataGenerator(rescale= 1./255,shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip= True)

#we dont do the data augmentation on test dataset
test_datagen = ImageDataGenerator(rescale = 1./255)

# Make sure you provide the same target size as initialied for the image size
#here we use class mode = categorical when we have more than two classes.
training_set = train_datagen.flow_from_directory("/content/drive/My Drive/CarBrand/Datasets/Train",
                                                target_size = (224, 224),
                                                batch_size = 32,
                                                class_mode = 'categorical')

test_set = test_datagen.flow_from_directory("/content/drive/My Drive/CarBrand/Datasets/Test",
                                           target_size = (224,224),
                                           batch_size = 32,
                                           class_mode = 'categorical')

#Here we can see that model is overfitting,so we d some hypermeter tuning.

#Now we fit the model.
r = model.fit(training_set,
             validation_data = test_set,
             epochs = 30,
             steps_per_epoch= len(training_set),
             validation_steps= len(test_set))



#plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

y_pred = model.predict(test_set)

y_pred

##Save model as a h5 file.
model.save("model_resnet50.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

new_model = load_model('model_resnet50.h5')

img = image.load_img("/content/drive/My Drive/CarBrand/Datasets/Test/audi/22.jpg", target_size = (224, 224))

x =  image.img_to_array(img)

x

x.shape

##scaling image dataset
x = x/255

x

x = np.expand_dims(x, axis = 0)

img_data = preprocess_input(x)

img_data.shape

img_data

new_model.predict(img_data)

a = np.argmax(new_model.predict(img_data), axis = 1)

if a == 0:
  print("Car is Audi")
elif a == 1:
  print("Car is Lamborghini")
else: 
  print("Car is Mercedes")

