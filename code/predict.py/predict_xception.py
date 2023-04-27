import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import matplotlib.pyplot as plt

# Set the path to your saved model
model_path = '/code/model/Xception.h5'

# Set the path to the image you want to classify
image_path = '/data/test/test1.jpg'

# Load the model
model = tf.keras.models.load_model(model_path)
validation_dir = '/data/dataset'

# Define the class labels
labels = ['Foi_Thong', 'Hang_Kra_Rog_Phu_Phan_ST1', 'Hang_Suea_Sakonnakhon_TT1', 'Kroeng_Krawia', 'Tanao_Si_Kan_Khaw_WA1', 'Tanao_Si_Kan_Dang_RD1']

target_size = (299, 299)

# Load the validation dataset
validation_generator = image.ImageDataGenerator(rescale=1./255).flow_from_directory(validation_dir, target_size=target_size, class_mode='categorical', batch_size=32)

# Evaluate the model on the validation dataset
score = model.evaluate(validation_generator)

# Print the accuracy of the model
print('Accuracy:', score[1])


# Load and preprocess the image
img = image.load_img(image_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Use the model to make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# Print the predicted class and its label
print('Cannabis Class:', labels[predicted_class])
prediction_values = {}
for i, label in enumerate(labels):
    prediction_values[label] = round(predictions[0][i], 2)

print('Prediction values:', prediction_values)

# Display the image
#plt.imshow(image_path)
#plt.axis('off')
#plt.show()
