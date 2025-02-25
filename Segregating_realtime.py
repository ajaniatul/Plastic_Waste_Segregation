import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model_path=r"C:\Users\MSI\Downloads\classification_model.h5"
model = load_model(model_path)

# Define the classes
class_labels = {0: 'Non-Recyclable', 1: 'Recyclable'}

# Function to preprocess the input frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (150, 150))  # Adjust size as per your model's input
    normalized_frame = resized_frame / 255.0  # Normalize the pixel values
    reshaped_frame = np.reshape(normalized_frame, (1, 150, 150, 3))  # Add batch dimension
    return reshaped_frame

# Initialize the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Assume we want to detect objects within a certain area in the frame
    height, width, _ = frame.shape
    roi = frame[int(height*0.3):height, int(width*0.3):width]  # Region of interest for detection

    # Preprocess the region of interest
    input_data = preprocess_frame(roi)
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    # Draw a bounding box and label
    if confidence > 0.95:  # Confidence threshold
        label = f"{class_labels[predicted_class]}: {confidence:.2f}"
        cv2.rectangle(frame, (int(width*0.3), int(height*0.3)), (width, height), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(width*0.3), int(height*0.3) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
    # If confidence is less than or equal to 0.5, label as "Recyclable"
        label = "Recyclable:" f"{confidence:.2f}"
        cv2.rectangle(frame, (int(width*0.3), int(height*0.3)), (width, height), (0, 0, 255), 2)  # Red box for low confidence
        cv2.putText(frame, label, (int(width*0.3), int(height*0.3) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with the bounding box
    cv2.imshow('Recyclable Object Detection', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
