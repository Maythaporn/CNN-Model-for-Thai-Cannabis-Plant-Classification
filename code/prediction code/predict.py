import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

selected_model = 'Xception'

image_path = input("Enter the file path of the image you want to classify: ")

print("\n  Please choose a model (default = Xception): ")
print("1. Xception")
print("2. InceptionV3")
print("3. ResNet50")
choose = input("Please Choose >> ")

if choose == "1":
    selected_model = 'Xception'

elif choose == "2":
    selected_model = "InceptionV3"

elif choose == "3":
    selected_model = "ResNet50"

else:
  print("ERROR")

# Load the selected model and make a prediction
model_path = f'/code/model/{selected_model}.h5'

# Load the model
model = tf.keras.models.load_model(model_path)

# Define the class labels
labels = ['Foi_Thong', 'Hang_Kra_Rog_Phu_Phan_ST1', 'Hang_Suea_Sakonnakhon_TT1', 'Kroeng_Krawia', 'Tanao_Si_Kan_Khaw_WA1', 'Tanao_Si_Kan_Dang_RD1']



# Load and preprocess the image
img = image.load_img(image_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Use the model to make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Print the predicted class and its label
print('Cannabis plant verieties:', labels[predicted_class])
prediction_values = {}
for i, label in enumerate(labels):
    prediction_values[label] = round(predictions[0][i], 2)

print('Prediction values:', prediction_values)