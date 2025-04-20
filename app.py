# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:38:13 2025

@author: LAB
"""
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pickle

#load model
with open('model.pkl','rb') as f:
    model = pickle.load(f)
  
#set title application
st.title("Image Classification with MobileNetV2 by Nway Nway Kay Khaing")   



#file upload
upload_file = st.file_uploader("Upload an image:", type=["jpg", "png", "jpeg"])

if upload_file is not None:
    img = Image.open(upload_file)
    st.image(image, caption='Upload Image')


#preprocess
    img = img.resize((224, 224)) 
    x = image.img_to_array(img)
    x = np.expand_dims(x,ais=0)
    x = preprocess_input(x)
    
    
    #pediction
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]
    
    #display prediction
    st.subheader = ("Prediction:")
    for i,pred in enumerate(top_preds):
        st.write(f"{i+1}. **{pred[1]}** — {round(pred[2]*100, 2)}%")
    