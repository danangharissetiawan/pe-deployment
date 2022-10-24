from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow as tf

import os
import numpy as np

model = load_model("../model/mobileNet3.h5")


img = image.load_img('static/images/uploads/5.jpg', target_size=(224, 224))
img = image.img_to_array(img)
img_arr = np.expand_dims(img, axis=0)
img_net = tf.keras.applications.mobilenet.preprocess_input(img_arr)
label = np.argmax(model.predict(img_net), axis=1)
label_names = ['cat', 'dear', 'dog', 'horse']

print(f"{label} = {label_names[label[0]]}")