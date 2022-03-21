import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50

img_size = (224, 224)
shape=(224, 224, 3)

def get_dataframe(folder):
    filenames = []
    labels = []
    for image in os.listdir('Food-5K/' + folder + '/'):
        filenames.append(image)
        labels.append(image.split('_')[0])
    filenames = np.array(filenames)
    labels = np.array(labels)
    df = pd.DataFrame()
    df['filename'] = filenames
    df['class'] = labels
    return df

train_data = get_dataframe('training')
test_data = get_dataframe('validation')

def process_images(folder, df):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
    )
    
    return datagen.flow_from_dataframe(
        df,
        directory='Food-5K/' + folder +'/',
        x_col='filename',
        y_col='class',
        class_mode='binary',
        target_size=img_size,
    )
    
train = process_images('training', train_data)
test = process_images('validation', test_data)

# Initialize the CNN - Res net is 50 layers deep
cnn = ResNet50(weights='imagenet', 
                             input_shape=shape,
                             include_top=False)
cnn.trainable = False

inputs = keras.Input(shape=shape)

# feature extraction layer
x = cnn(inputs, training=False)

# pooling layer
x = keras.layers.GlobalAveragePooling2D()(x)

targets = keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, targets)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train, epochs=10, validation_data=test)

model.save("cnn_model.h5")