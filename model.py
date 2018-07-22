import numpy as np
import csv
import cv2

lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
       
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    #print(current_path)
    image = cv2.imread(current_path)
    images.append(image)
    image = cv2.flip(image,0)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
'''    
lines1 = []
with open('data2/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line1 in reader:
        lines1.append(line1)
for line in lines1:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data2/IMG/' + filename
    #print(current_path)
    image = cv2.imread(current_path)
    images.append(image)
    image = cv2.flip(image,0)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    measurements.append(-measurement)
'''
print(str(np.shape(images)))
X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,5,5,subsample=(2,2),activation='relu'))
#model.add(Convolution2D(64,5,5,subsample=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.25, shuffle=True, nb_epoch=6)

model.save('model.h5')
