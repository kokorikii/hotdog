import streamlit as st
import plotly.express as px
import numpy as np
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import load_model
import time
from tempfile import NamedTemporaryFile
from keras_preprocessing.image import load_img, img_to_array

# streamlit run hotdog.py

st.title('Hotdog or Not, Dawg')
st.header('Is that thing a hotdog?')
st.markdown("---")


picture = st.file_uploader('Give it the dog')
temp_file = NamedTemporaryFile(delete=False)

if picture:
    temp_file.write(picture.getvalue())
    hotdog = load_img(temp_file.name)
    rgb_im = hotdog.convert('RGB')
    hotdog_arr = tf.keras.preprocessing.image.img_to_array(rgb_im) / 255
    resized_hotdog = tf.image.resize(hotdog_arr, (256, 256))
    hotdog_array = np.array(resized_hotdog).reshape(1,256,256,3)
    loaded_model = tf.keras.models.load_model('best_model_file.h5')
    prediction = loaded_model.predict(hotdog_array)[0][0]
    
    if prediction >= .5:
        st.header(f"I'm {round((prediction * 100), 2)}% sure that's a hotdog, dawg. Data don't lie.")
    else:
        st.header(f"I'm {round(((1 - prediction) * 100), 2)}% sure that's NOT a hotdog, dawg. Data don't lie.")
    st.image(hotdog)
