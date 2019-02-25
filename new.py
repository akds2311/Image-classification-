# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL
from glob import glob

# loading the directories
training_dir = "./assig2/training"
validation_dir = './assig2/training'
# test_dir = '../input/fruits/fruits-360_dataset/fruits-360/test-multiple_fruits/'
#

# useful for getting number of files
image_files = glob(training_dir + '/*/*.jpg')
# valid_image_files = glob(validation_dir + '/*/*.jpg')

# getting the number of classes i.e. type of fruits
folders = glob(training_dir + '/*')
num_classes = len(folders)

# # this will copy the pretrained weights to our kernel
# !mkdir ~/.keras
# !mkdir ~/.keras/models
# !cp ../input/keras-pretrained-models/*notop* ~/.keras/models/
# !cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/

# importing the libraries
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
#from keras.preprocessing import image

IMAGE_SIZE = [224, 224]  # we will keep the image size as (64,64). You can increase the size for better results.

# loading the weights of VGG16 without the top layer. These weights are trained on Imagenet dataset.
conv_base = VGG16(input_shape = IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)  # input_shape = (64,64,3) as required by VGG

# this will exclude the initial layers from training phase as there are already been trained.
for layer in conv_base.layers:
    layer.trainable = False

x = Flatten()(conv_base.output)
#x = Dense(128, activation = 'relu')(x)   # we can add a new fully connected layer but it will increase the execution time.
x = Dense(num_classes, activation = 'softmax')(x)  # adding the output layer with softmax function as this is a multi label classification problem.

model = Model(inputs = conv_base.input, outputs = x)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Image Augmentation

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input

training_datagen = ImageDataGenerator(
                                    rescale=1./255,   # all pixel values will be between 0 an 1
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    preprocessing_function=preprocess_input, validation_split=0.1993)

# validation_datagen = ImageDataGenerator(rescale = 1./255, preprocessing_function=preprocess_input)

training_generator = training_datagen.flow_from_directory(training_dir, target_size = IMAGE_SIZE, batch_size = 20, class_mode = 'categorical',subset='training')
validation_generator = training_datagen.flow_from_directory(validation_dir, target_size = IMAGE_SIZE, batch_size = 20, class_mode = 'categorical', subset='validation')
training_generator.class_indices
training_images = 1943
validation_images =480

    history = model.fit_generator(training_generator,
                       steps_per_epoch = 1943,  # this should be equal to total number of images in training set. But to speed up the execution, I am only using 10000 images. Change this for better results.
                       epochs = 10,  # change this for better results
                       validation_data = validation_generator,
                       validation_steps = 480)  # this should be equal to total number of images in validation set.
