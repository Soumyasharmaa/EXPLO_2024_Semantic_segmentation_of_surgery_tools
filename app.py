
from re import I
from tensorflow.keras import backend as K
import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import cv2
#from streamlit import caching
from PIL import Image
#from preprocessing_images import *

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
##########################################################
import requests
import urllib.request
import os
import tempfile

def download_model_weights(url):
    response = requests.get(url)
    if response.status_code == 200:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(response.content)
        temp_file.close()
        return temp_file.name
    else:
        raise Exception(f"Failed to download model weights: {response.status_code}")


model_weights_url = 'https://drive.google.com/drive/folders/1xptXoHHBXtoaQRIQKZr1I7vABukGlTjh?usp=drive_link'

# Download and save model weights
model_weights_file = download_model_weights(model_weights_url)

# Load model from the saved file
try:
    model = load_model(model_weights_file, custom_objects={
        'jaccard_distance_loss': jaccard_distance_loss,
        'dice_coef': dice_coef,
        'iou_metric': iou_metric
    })
    st.write("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
##################################################################
def jaccard_distance_loss(y_true, y_pred,smooth = 100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

# Define the Intersection over Union (IoU) metric
def iou_metric(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true, axis=-1) + K.sum(y_pred, axis=-1) - intersection
    return (intersection + smooth) / (union + smooth)

def make_prediction(model,image,shape):
    img = img_to_array(load_img(image,target_size=shape))
    img = np.expand_dims(img,axis=0)/255.
    mask = model.predict(img)

    mask = (mask[0] > 0.5)*1
#     print(np.unique(mask,return_counts=True))
    if model == VGG16:
      mask = np.reshape(mask,(960,960))
      return mask
    mask = np.reshape(mask,(224,224))
    return mask

# Load the saved model
# with custom_object_scope({'jaccard_distance_loss': jaccard_distance_loss,'dice_coef': dice_coef}):
#     model = load_model('https://drive.google.com/file/d/1-hmf_Fd3P4xAIFXt768GEefN85tzAWQT/view?usp=drive_link')  # Replace with your model file path


# # Load the saved model
# model = load_model(model_weights_file, custom_objects={'jaccard_distance_loss': jaccard_distance_loss, 'dice_coef': dice_coef, 'iou_metric': iou_metric})


######################################### vGG 16
from skimage.io import imread
from skimage.transform import resize

# function to predict result
def predict_image(img_path, model):
    H = 480
    W = 480
    num_classes = 4

    img = img_to_array(load_img(img_path))
    img = img[:480, :480, :]
    img = img / 255.0
    img = img.astype(np.float32)

    ## Read mask
    # mask = imread(mask_path, as_gray = True)
    # mask = mask[:480, :480]

    ## Prediction
    pred_mask = model.predict(np.expand_dims(img, axis=0))
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[0]


    # calculating IOU score
    # inter = np.logical_and(mask, pred_mask)
    # union = np.logical_or(mask, pred_mask)

    # iou = inter.sum() / union.sum()

    return img, pred_mask

import tensorflow as tf

def iou_score(y_true, y_pred):
    inter = np.logical_and(mask, pred_mask)
    union = np.logical_or(mask, pred_mask)

    iou = inter.sum() / union.sum()
    return iou
def f1_score(y_true, y_pred):
  # Compute the F1 score using sklearn's f1_score function
  # You may need to adjust the parameters depending on your task
  return f1_score(y_true, y_pred)
# Define custom objects dictionary
custom_objects = {'BatchNormalization': tf.keras.layers.BatchNormalization}
# Register the custom metric function
tf.keras.utils.get_custom_objects()['iou_score'] = iou_score
tf.keras.utils.get_custom_objects()['f1-score'] = f1_score
# Load the model with custom objects
VGG16 = tf.keras.models.load_model('/content/drive/MyDrive/Explo_2024_sem4/VGG16.h5', custom_objects=custom_objects)

st.title('Exploratory Project : Surgery Tools segmentation application ')

st.markdown("***")


# st.subheader('Choose the model to be used for segmentation')
# option = st.radio('',('U_Net', 'VGG 16'))
# st.write('You selected:', option)
option = 'U_Net'

if option == 'U_Net':
    st.subheader('Upload the image to be segmented fo surgery tools')
    uploaded_file = st.file_uploader(' ',accept_multiple_files = False)

    if uploaded_file is not None:
        # Perform your Manupilations (In my Case applying Filters)
        image = uploaded_file
        img = img_to_array(load_img(image))
        st.write("Image Uploaded Successfully")
        #st.write(plt.imshow(img/255.))
        st.write("shape of the input image is : " + str(img.shape))
        #img = load_preprocess_image(str(img))

        st.image(img/255.)
        mask = make_prediction(model,image,(224,224,3))
        mask2 = cv2.merge([mask,mask,mask]).astype('float32')
        #st.write(img.shape,mask2.shape)
        mask2 = cv2.resize(mask2,(img.shape[1],img.shape[0]))
        # print(mask.shape)
        st.image(mask2)
        h,w = img.shape[:2]
        mask_resized = cv2.resize(np.uint8(mask*1),(w,h))
        mask_resized = mask_resized != 0
        #print(np.unique(mask_resized,return_counts=True))
        segment = np.zeros((h,w,3))
        segment[:,:,0] = img[:,:,0]*mask_resized
        segment[:,:,1] = img[:,:,1]*mask_resized
        segment[:,:,2] = img[:,:,2]*mask_resized
        segment[np.where((segment == [0,0,0]).all(axis=2))] = [0,0,0]
        #img[np.where((img==[255,255,255]).all(axis=2))] = [0,0,0];
        #plt.figure(figsize=(8,8))
        st.image(segment/255.)
      
    else:
        st.write("Make sure you image is in TIF/JPG/PNG Format.")

elif option == 'VGG 16':
    st.subheader('Upload the image to be segmented fo surgery tools')
    uploaded_file = st.file_uploader(' ',accept_multiple_files = False)
    if uploaded_file is not None:
      image = uploaded_file
      img, pred_mask= predict_image(image,VGG16)
      st.image(img/255.)
      st.image(pred_mask)

    else:
      st.write("Make sure you image is in TIF/JPG/PNG Format.")


st.markdown("***")

#st.write(' Try again with different inputs')

result = st.button('Try again')
if result:
	
	uploaded_file = st.empty()
	predict_button = st.empty()
	#caching.clear_cache()
