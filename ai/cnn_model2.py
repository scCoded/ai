import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from sklearn.metrics import classification_report

img_size = (224, 224)
shape=(224, 224, 3)

def get_dataframe(folder):
    filenames = []
    labels = []
    path = 'Food-5K/' + folder + '/'
    for image in os.listdir(path):
        filenames.append(image)
        labels.append(image.split('_')[0])
    filenames = np.array(filenames)
    labels = np.array(labels)
    df = pd.DataFrame()
    df['filename'] = filenames
    df['class'] = labels
    return df

train_images = get_dataframe('training')
test_images = get_dataframe('validation')

def process_images(folder, df):
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
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
    
train_generator = process_images('training', train_images)
test_generator = process_images('validation', test_images)

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

outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=1, validation_data=test_generator)

#evaluate
"""
y_true = []
y_pred = []

for image in os.listdir('Food-5K/evaluation'):
    img = Image.open('Food-5K/evaluation/' + image)
    img = img.resize(img_size)
    img = np.array(img)
    img = np.expand_dims(img, 0)
    img = tf.cast(img, tf.float32)
    y_true.append(int(image.split('_')[0]))
    y_pred.append(1 if model.predict(img) > 0.5 else 0)
    
print(classification_report(y_true, y_pred))
"""
model.save("cnn_model2.h5")
