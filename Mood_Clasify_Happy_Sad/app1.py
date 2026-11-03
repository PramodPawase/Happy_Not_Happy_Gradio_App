
# ==================================================
# Import Libraries
# ==================================================

import os
import numpy as np
import tensorflow as tf
import gradio as gr

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ==================================================
# Project Paths
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "training")
VALIDATION_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "testing")

# ==================================================
# Data Generators
# ==================================================

train_datagen = ImageDataGenerator(
    rescale=1 / 255
)

validation_datagen = ImageDataGenerator(
    rescale=1 / 255
)

test_datagen = ImageDataGenerator(
    rescale=1 / 255
)

# ==================================================
# Dataset Loading
# ==================================================

train_dataset = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(200, 200),
    batch_size=32,
    class_mode="binary"
)

validation_dataset = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(200, 200),
    batch_size=32,
    class_mode="binary"
)

test_dataset = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(200, 200),
    batch_size=32,
    class_mode="binary",
    shuffle=False
)

print("\nClass Mapping:")
print(train_dataset.class_indices)

print("\nTotal Training Images:")
print(len(train_dataset.filenames))

# ==================================================
# CNN Model
# ==================================================

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        input_shape=(200, 200, 3)
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        512,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )

])

model.summary()

# ==================================================
# Compile Model
# ==================================================

model.compile(
    optimizer=tf.keras.optimizers.RMSprop(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ==================================================
# Train Model
# ==================================================

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# ==================================================
# Evaluate Model
# ==================================================

test_loss, test_accuracy = model.evaluate(
    test_dataset
)

print(f"\nTest Accuracy: {test_accuracy:.4f}")

# ==================================================
# Save Model
# ==================================================

model.save("happy_sad_model.h5")

print("\nModel saved successfully.")

# ==================================================
# Prediction Function
# ==================================================

def predict_mood(uploaded_image):
    """
    Predict Happy or Sad
    """

    img = uploaded_image.resize((200, 200))

    img_array = np.array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    if prediction < 0.5:
        return "😊 Happy"

    return "😔 Sad"


# ==================================================
# Gradio Interface
# ==================================================

iface = gr.Interface(
    fn=predict_mood,
    inputs=gr.Image(
        type="pil",
        label="Upload Face Image"
    ),
    outputs=gr.Textbox(
        label="Prediction"
    ),
    title="Happy vs Sad Classifier",
    description="""
    Upload an image and the CNN model will
    classify whether the person is Happy or Sad.
    """
)

# ==================================================
# Launch App
# ==================================================

if __name__ == "__main__":
    iface.launch()