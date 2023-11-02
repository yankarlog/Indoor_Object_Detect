import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

# Defining labels
labels = ["door", "openedDoor", "cabinetDoor", "refrigeratorDoor", "window", "chair", "table", "cabinet", "sofa", "pole"]

# Backstory
st.title("Object Identification AI")

model = tf.keras.models.load_model("yorha1.h5")
st.title("Object Identification AI")
# Create a VideoCapture object for camera input
cap = cv2.VideoCapture(0)  # 0 for the default camera (you can change this if you have multiple cameras)

frame_counter = 0  # Initialize a frame counter
model_frequency = 5  # Process a frame every 5 frames

# Function to capture and display frames
while True:
    ret, frame = cap.read()  # Read a frame from the camera

    if not ret:
        st.error("Error capturing a frame. Check your camera connection.")
        break

    frame_counter += 1  # Increment the frame counter

    # Display the frame in Streamlit
    st.image(frame, channels="BGR", use_column_width=True)

    # Process an image every 5 frames
    if frame_counter % model_frequency == 0:
        # Here, you can insert your code to process the frame with your model
        # For example, you can resize the frame and perform object detection
        
        image = cv2.resize(frame, (150, 150), interpolation=cv2.INTER_LINEAR)

        if image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = np.expand_dims(image, axis=0)

        st.image(image, caption="Resized Image (150x150)", use_column_width=True)

        # Make a prediction using the loaded model
        predictions = model.predict(image)
        top_prediction = labels[np.argmax(predictions)]

        st.write(f"Predicted Object: {top_prediction}")

# Release the VideoCapture object and close the Streamlit app
cap.release()
st.stop()

