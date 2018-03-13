import os
import csv

import cv2
import numpy as np
import sklearn
#import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Lambda, Dropout, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
#from keras.backend import tf as ktf

samples = []
with open('../P3_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
print(np.shape(samples))

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
print(np.shape(train_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../P3_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #Adding flipped image on the center image
                images.append(cv2.flip(center_image,1))
                angles.append(center_angle*-1.0)
                
                name1 = '../P3_data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name1)
                images.append(left_image)
                angles.append(center_angle+0.2)
                
                name2 = '../P3_data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name2)
                images.append(right_image)
                angles.append(center_angle-0.2)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
# each call to generator will return 4 times the size of batch_size of samples,
#because, original+ flipped images + left image + right image
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)


#ch, row, col = 3, 80, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
#model.add(... finish defining the rest of your model architecture here ...)


'''
lines = []
with open('../P3_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
	for i in range(3):
		source_path = line[i]
		filename = source_path.split('/')[-1] # choose the last token as filename
		current_path = '../P3_data/IMG/' + filename
		image = cv2.imread(current_path)
		#print(current_path)
		images.append(image)
		if i==0: # center image
			st_ang = float(line[3])
		elif i==1: # for left image, add a correction factor to steering angle
			st_ang = float(line[3]) +0.2
		elif i==2: # for right image, subtract a correction factor from steering angle
			st_ang = float(line[3]) - 0.2
		else:
			st_ang = float(line[3])
			
		measurements.append(float(line[3])) #Steering Angle
	
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	augmented_images.append(cv2.flip(image,1))
	augmented_measurements.append(measurement*-1.0)
	
#keras need numpy array as input
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)
print(np.shape(X_train))

#resized_X = tf.image.resize_images(X_train,(32,32))
#resized_X = np.array(resized_X)

model = Sequential()
#model.add(Lambda(lambda image: ktf.image.resize_images(image, (32, 32)),input_shape = X_train.shape[1:]))
#model.add(Lambda (lambda x:x / 255.0 - 0.5))
model.add(Lambda (lambda x:x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
'''


'''
#...........LeNet architecture.........................................
model.add(Convolution2D(6,5,5, border_mode='valid', activation="relu"))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.75))
#model.add(Activation('relu'))

model.add(Convolution2D(16,5,5, border_mode='valid', activation="relu"))
model.add(MaxPooling2D((2,2)))
#model.add(Dropout(0.75))
#model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dense(84))
model.add(Activation('relu'))
model.add(Dense(1))
#no activation function, so regression
#.........End LeNet architecture........................................
'''

#.....architecture used by autonomous driving group in nvidia...........
model.add(Convolution2D(24,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Convolution2D(64,3,3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#.......................................................................
'''
model.compile(loss = 'mse', optimizer ='adam')
model.fit(x = X_train, y= y_train, nb_epoch=3, validation_split =0.2, shuffle = True,verbose = 1)
'''
model.compile(loss='mse', optimizer='adam')
# samples_per_epoch and nb_val_samples is set to 4 times the train and validation samples
# size, because generator returns 4 * batch_size of samples in each batch
model.fit_generator(train_generator, samples_per_epoch= 4*len(train_samples),\
validation_data=validation_generator, nb_val_samples=4*len(validation_samples), \
nb_epoch=5, verbose = 1)

model.save('model.h5')



