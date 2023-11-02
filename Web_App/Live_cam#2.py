import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Defining vars
labels = ["closed door", "Door", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "sofa", "pole"]
frame_counter = 0  # Initialize a frame counter
model_frequency = 5  # use the img every 5 frames
model = tf.keras.models.load_model("yorha1.h5")
cap = cv2.VideoCapture(0)  #0 = first cam


#page build
st.title("Object Identification AI")


#Get and show
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        st.error("Check thy current camera because, it appaers that we art getting thy signals from a camera situated in another dimension")
        break

    frame_counter += 1  # Increment the frame counter

    #Get the img from live feed
    if frame_counter % model_frequency == 0:

        #make sure it's appropiate dimentions
        image = cv2.resize(frame, (150, 150), )#interpolation=cv2.INER_LINEAR

        #Recomendation from GPT check
        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.expand_dims(image, axis=0)

        st.image(image, caption="Resized Image (150x150)", use_column_width=True)

        #predictions
        predictions = model.predict(image)

        top_prediction = np.argmax(predictions)

        st.write(f"It's a .... {top_prediction}")



# Release the VideoCapture object and close the Streamlit app
cap.release()
st.stop()
