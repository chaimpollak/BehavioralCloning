# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 21:22:58 2016

@author: Chaim Pollak
"""
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D,MaxPooling2D

        
def get_train_val_test_data(p_val= 0.2, p_test= 0.25):
    """
    This function will return train, test and validation dataframes from the driving_log.csv file.
    @param p_val - the percentage of the data to be used for the validation dataset
    @param p_test - the percentage of the data to be used for the test dataset
    
    The method assumes the driving_log.csv file is located in the current directory.
    """
    df = pd.read_csv('driving_log.csv')   
    new_df = pd.DataFrame( columns=['steering','img'])
    
    ## get the center images and store it in the new dataframe
    temp_df = df[['steering', 'center']]
    temp_df.columns = ['steering','img']
    new_df = new_df.append(temp_df)

    ### get the left side images and add it to the dataframe
    temp_df = df[['steering', 'left']]
    temp_df.columns = ['steering','img']
    temp_df.steering = temp_df.steering.apply(lambda x: x+(3./25.0)) ##=> add 0.12 to the steering angle for the left image
    temp_df.img = temp_df.img.apply(lambda x: x.strip())
    new_df = new_df.append(temp_df)

    ### get the right side images and add it to the dataframe
    temp_df = df[['steering', 'right']]
    temp_df.columns = ['steering','img']
    temp_df.steering = temp_df.steering.apply(lambda x: x-(3.0/25.0)) ##=> negate 0.12 to the steering angle for the right image
    temp_df.img = temp_df.img.apply(lambda x: x.strip())
    new_df = new_df.append(temp_df)
    new_df = new_df.reset_index(drop=True)
    
    ## shuffle the data
    new_df = new_df.iloc[np.random.permutation(len(new_df))]

    ## split the data into train, validation and test dataframes
    msk = np.random.rand(len(new_df)) > (p_val+p_test)
    train, val_test = new_df[msk], new_df[~msk]

    train = train.reset_index(drop=True)
    val_test = val_test.reset_index(drop=True)

    msk = np.random.rand(len(val_test)) < (p_val/(p_test+p_val))
    val, test = val_test[msk],val_test[~msk]

    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)
    return train, val, test
    
def process_image(img):
    """
    This function will do the following image manipulation in order to help to model predict
    the steering angle.
    - crop the image by removing the top 40 pixels of the image (since it is useless for the model)
    - resize the image to half its size - (the less data the quicker the model will find the relative  information)
    - change the image from RGB to YUV - (this really helps smoothing the image so the model has less info to process)
    """
    img = img[40:,:,:] ## crop the image
    img = cv2.resize(img,(160,60), interpolation = cv2.INTER_CUBIC) ## shrink the image size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) ## change to YUV
    img = img.astype('float32')
    return img

def generate_image_from_dataframe(df, batch_size=256):
    """
    @param batch_size - the size of x and y values to return
    @return a tuple with x and y values the length of batch_size where x is the 
    processed image and y is the steering angle.
    """
    xx = np.zeros((batch_size, 60, 160, 3))
    yy = np.zeros(batch_size)
    i = 0
    while True:
        for ii in range(batch_size):
            x = plt.imread(df['img'].loc[i]) 
            x = process_image(x)
            y= np.array(float(df['steering'].loc[i]))
            xx[ii] = x.reshape((1,) + x.shape)
            yy[ii] = y.reshape((1,) + y.shape)
            i += 1
            if i == len(df):
                i = 0
        yield xx, yy
        
   
def get_model(input_shape):
    """
    This function will return the model to be used to predict the steering angle.
    The model has 6 conv-nets with max-pooling and dropouts between every 2 layers
    and 5 neural layers (the layer size is based off the nvidia model).
    The model will first normalize the data and then run it through the network.
    """
    
    pool_size = (2, 2)
    stride_size = (2,2)

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(3, 5, 5,border_mode='same',subsample=stride_size))
    model.add(Convolution2D(24, 5, 5,border_mode='same',subsample=stride_size))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(.5))
    model.add(Convolution2D(36, 5, 5,border_mode='same',subsample=stride_size))
    model.add(Convolution2D(48, 3, 3,border_mode='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(.5))
    model.add(Convolution2D(64, 3, 3,border_mode='same'))
    model.add(Convolution2D(128, 3, 3,border_mode='same'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Flatten())
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1164))
    model.add(ELU())
    model.add(Dense(100))
    model.add(ELU())
    model.add(Dense(50))
    model.add(ELU())
    model.add(Dense(10))
    model.add(ELU())
    model.add(Dense(1))
    return model

         
train, test, val = get_train_val_test_data()
model = get_model( next(generate_image_from_dataframe(test,batch_size=1))[0].shape[1:])
model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])

history = model.fit_generator(generate_image_from_dataframe(train), samples_per_epoch=len(train), nb_epoch=10, validation_data=generate_image_from_dataframe(val),nb_val_samples=len(val),verbose=1)
prediction = model.evaluate_generator(generate_image_from_dataframe(test), val_samples=len(test))

print("loss value on training data:", history.history['loss'][-1])
print("loss value on validation data:", history.history['val_loss'][-1])
print("loss value on test data:" ,prediction[0])

## Save the model and weights

model.save_weights('model.h5')
import json
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)