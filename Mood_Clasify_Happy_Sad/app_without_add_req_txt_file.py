from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os


img = image.load_img('/content/drive/MyDrive/CNN-Happy-Sad/training/happy/PublicTest_98770281.jpg')

i1 = cv2.imread('/content/drive/MyDrive//CNN-Happy-Sad/training/happy/PublicTest_98770281.jpg')
#i1

train=ImageDataGenerator(rescale=1/255)
validation=ImageDataGenerator(rescale=1/255)


train_dataset = train.flow_from_directory('/content/drive/MyDrive/CNN-Happy-Sad/training/',
                                         target_size = (200,200),
                                         batch_size = 3,
                                         class_mode = 'binary')
validataion_dataset = validation.flow_from_directory('/content/drive/MyDrive/CNN-Happy-Sad/validation',
                                          target_size = (200,200),
                                          batch_size = 3,
                                          class_mode = 'binary')


train_dataset.class_indices


train_dataset.classes

# now we are applying maxpooling 
model = tf.keras.models.Sequential([ tf.keras.layers.Conv2D(16,(3,3),activation = 'relu',
                                                            input_shape = (200,200,3)),
                                    tf.keras.layers.MaxPool2D(2,2), #3 filtr we applied hear
                                    #
                                    tf.keras.layers.Conv2D(32,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    #
                                    tf.keras.layers.Conv2D(64,(3,3),activation = 'relu'),
                                    tf.keras.layers.MaxPool2D(2,2),
                                    ##
                                    tf.keras.layers.Flatten(),
                                    ##
                                    tf.keras.layers.Dense(512, activation = 'relu'),
                                    #
                                    tf.keras.layers.Dense(1,activation= 'sigmoid')
                                    ]
                                    )

model.compile(loss='binary_crossentropy',
              optimizer = tf.keras.optimizers.RMSprop(learning_rate = 0.001),
              metrics = ['accuracy']
              )

model_fit = model.fit(train_dataset,
                     steps_per_epoch = 3,
                     epochs = 10)

import gradio as gr
from PIL import Image
import numpy as np

def predict_mood(image):
    # Resize the image to (200, 200) as expected by the model
    img = image.resize((200, 200))
    # Convert the image to a numpy array
    x = np.array(img)
    # Expand dimensions to create a batch of 1 image
    x = np.expand_dims(x, axis=0)
    # Normalize the image (the model was trained with rescale=1/255)
    x = x / 255.0
    # Make prediction
    val = model.predict(x)[0][0]
    # Interpret the prediction
    if val < 0.5:
        return 'Happy'
    else:
        return 'Not Happy'


iface = gr.Interface(fn=predict_mood,
                     inputs=gr.Image(type="pil", label="Upload an image"),
                     outputs=gr.Text(label="Predicted Mood"),
                     title="Mood Classification (Happy/Not Happy)",
                     description="Upload an image to classify if the person is happy or not happy.")
iface.launch()    