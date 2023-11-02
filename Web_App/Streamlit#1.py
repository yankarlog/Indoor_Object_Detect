#init libraries
import streamlit as w
import numpy as np
from joblib import load
import cv2
import pandas as pd
import numpy as np
import os
import h5py
import tensorflow as tf

# Defining labels
pout = ["door", "openedDoor", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "sofa", "pole"]

# Backstory
w.title("Helcome to the object identification AI developed")
# w.write(pout[0], "afg")


# quanta=w.camera_input()

h5f = h5py.File("yorha1.h5", 'r')

# Load the model architecture and weights
model = tf.keras.models.load_model("yorha1.h5")

# Close the HDF5 file
h5f.close()

f = w.file_uploader("Upload Image")

if f is not None:
    byte = np.asarray(bytearray(f.read()), dtype=np.uint8)
    img = cv2.imdecode(byte, 1)
    w.image(img, channels="BGR")
    
    # img=cv2.resize(img, (150,150), interpolation=cv2.INTER_LINEAR)
    image = cv2.imdecode(np.fromstring(f.read(), np.uint8), 1)

    # Resize the image to 150x150 pixels
    resized_image = cv2.resize(image, (150,150), interpolation=cv2.INTER_LINEAR)

    


    # Convert the image to RGB format (3 channels)
    if resized_image.shape[2] == 1:  # Convert grayscale to RGB
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_GRAY2RGB)
    resized_image = np.expand_dims(resized_image, axis=0)
    # Display the resized image
    w.image(resized_image, caption="Resized Image (150x150)", use_column_width=True)

    # Make a prediction using the loaded model
    result = model.predict([resized_image])  # Pass the image as a list or array
    w.write(f"It looks like {pout[result[0].argmax()]}")
    w.write(f"result: \n{result}")