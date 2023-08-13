import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
# Load the pre-trained model from the .h5 file
model = load_model('model_v1.h5')

# Create a Tkinter application
app = tk.Tk()
app.title("COVID-19 Detection")

# Set window size and position
app.geometry("500x500")
app.resizable(False, False)

# Define a function to make predictions
def predict_covid(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

    prediction = model.predict(img_array)
    print("Prediction array:", prediction)

    if prediction[0][1] > 0.5:  # Compare the probability of "COVID-19" class
        app.configure(bg="#C0392B")
        return "COVID-19"
    else:
        app.configure(bg="#3498DB")
        return "Normal"

# Define a function to handle button click event
def on_upload_button_click():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_covid(file_path)
        result_label.config(text="Prediction: " + result, fg="#1E8449" if result == "Normal" else "#C0392B")
        display_image(file_path)

# Function to display the uploaded image
def display_image(image_path):
    img = Image.open(image_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

# Widgets and layout
title_label = tk.Label(app, text="COVID-19 Detection", font=("Helvetica", 24), fg="black")
title_label.pack(pady=10)

upload_button = tk.Button(app, text="Upload Image", command=on_upload_button_click, font=("Helvetica", 14), bg="#3498DB", fg="white")
upload_button.pack(pady=20)

image_label = tk.Label(app)
image_label.pack()

result_label = tk.Label(app, text="", font=("Helvetica", 16), wraplength=400)
result_label.pack(pady=20)

# Start the Tkinter event loop
app.mainloop()