#Loading Required Modules
import csv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Flatten,Dense,Lambda,Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D,MaxPooling2D
import sklearn
from sklearn.model_selection import train_test_split
import random as rand
#Folders Will Store Name of Folders That Contain Training Images
parent_folder='data'
folders=[]
folder_list=os.listdir(parent_folder+'/')
#Correction that later will be applied to Left and Right Camera Measurement Values
correction=0.2
epochs=8
batch_size=32
#Reads lines in csv files in different folders inside parent folder
lines=[]
for folder in folder_list:
  #Opening CSV File in Each Folder and Reading each Line and Storing in Lines[]
  with open('./'+parent_folder+'/'+folder+'/driving_log.csv') as csvfile:
      reader =csv.reader(csvfile)
      for line in reader:
        line.append(folder)
        lines.append(line)
# Each line has 3 images(center, left and right) and each image also will have a flip so
# number of lines*6 is total number of images
print('Number of Lines',len(lines))
print('Total Number of images:',6*len(lines))
#Splitting data to train and test data. 20% of images will be test images
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
#Reading each Lines and Extracting The Path For Center, Left and Right Images and reading
#them. Also adding flipped images and measurements. All happens in generator to make sure
# there will be no problem with memory.
def generator(lines, batch_size=batch_size):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        rand.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images=[]
            measurements=[]
            for line in batch_samples:
                main_folder=line[-1]
                measurement=float(line[3])
                flipped_measurement=(measurement)*-1.0
                for i in range(0,3):
                   source_path=line[i]
                   tokens=source_path.split('/')
                   filename=tokens[-1]
                   local_path='./'+parent_folder+'/'+main_folder+"/IMG/"+filename
                   #Reading Image and measurement and Appending to Images[] and #measurements[]
                   image=cv2.imread(local_path)
                   images.append(image)
                   #Flipping image and measurement and adding correction
                   flipped_image=cv2.flip(image,1)
                   images.append(flipped_image)
                   if i==0:
                      measurements.append(measurement)
                      measurements.append(flipped_measurement)
                   elif i==1:
                      measurements.append(measurement+correction)
                      measurements.append(flipped_measurement+correction)
                   else:
                      measurements.append(measurement-correction)
                      measurements.append(flipped_measurement-correction)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)
#Defining Keras model, Modified NVIDIDA
model=Sequential()
#Adding normalization and cropping layers
model.add(Lambda(lambda x:x/255.0-0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((75,25),(20,20))))
#Main model
model.add(Convolution2D(16, 3,3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
#Using Adam Optimizer and MSE Loss Function For Regression 
model.compile(optimizer='adam',loss='mse')
#Training The Model,20% of Data is Allocated for Validation
#It is important to notice that there are 6 images per line is read in generator, so in the
# fit_generator we must consider 6*len(train_samples) as total number of samples
model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples), validation_data=validation_generator,
            nb_val_samples=6*len(validation_samples), nb_epoch=epochs)
#Saving The Model
model.save('model.h5')
