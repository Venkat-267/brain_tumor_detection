import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import ipywidgets as widgets
import io
from PIL import Image
from IPython.display import display,clear_output

# Load your trained AI model
model = tf.keras.models.load_model('effnet.h5')

def img_pred(image):
    opencvImage = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(opencvImage, (150, 150))
    img = img.reshape(1, 150, 150, 3)
    p = model.predict(img)
    p = np.argmax(p, axis=1)[0]

    if p == 0:
        return 'Glioma Tumor'
    elif p == 1:
        return 'No Tumor'
    elif p == 2:
        return 'Meningioma Tumor'
    else:
        return 'Pituitary Tumor'


def main():
    st.title('Brain Tumor Detection')
    st.write('Upload a Brain MRI image and the AI model will predict whether there is a tumor or not.')

    uploaded_image = st.file_uploader('Choose a Brain MRI image...', type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded MRI Image', use_column_width=True)

        # Make prediction using the AI model
        prediction = img_pred(image)

        # Show the result to the user
        st.write(f'Prediction: {prediction}')

if __name__ == '__main__':
    main()
