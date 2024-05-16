import datetime
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Load the trained model
model = load_model('Sign_Language_Model.h5')

# Instantiate the LabelEncoder and load the label encoder classes
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('classes.npy')

# Function to check if the current time is valid
def is_valid_time():
    current_time = datetime.datetime.now().time()
    start_time = datetime.time(18, 0)  # 6 PM
    end_time = datetime.time(22, 0)    # 10 PM
    return start_time <= current_time <= end_time

# Function to preprocess a single image
def preprocess_image(image_path, img_size=(64, 64)):
    image = load_img(image_path, target_size=img_size)
    image = img_to_array(image)
    image = image / 255.0
    return image

# Define the prediction function for images
def predict_sign_language(image_path):
    if not is_valid_time():
        messagebox.showwarning("Time Restriction", "Predictions are only allowed between 6 PM and 10 PM")
        return

    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)
    result = label_encoder.inverse_transform([predicted_class])[0]
    result_label.config(text=f"Prediction: {result}")
    print(result)

# Function to open a file dialog and get the image path
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img

        # Predict the sign language in the image
        predict_sign_language(file_path)

# Set up the GUI
root = tk.Tk()
root.title("Sign Language Recognition")

# Create a button to upload an image
upload_btn = tk.Button(root, text="Upload Image", command=upload_image)
upload_btn.pack(pady=20)

# Create a label to display the selected image
panel = tk.Label(root)
panel.pack(pady=20)

# Create a label to display the prediction result
result_label = tk.Label(root, text="Prediction: ", font=("Helvetica", 16))
result_label.pack(pady=20)

# Run the GUI loop
root.mainloop()