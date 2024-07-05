
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

def make_prediction(image,shape):
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




# # Load the saved model
# with custom_object_scope({'jaccard_distance_loss': jaccard_distance_loss,'dice_coef': dice_coef}):
#     model = load_model('Soumyasharmaa/EXPLO_2024_Semantic_segmentation_of_surgery_tools/best_model_final.h5')  # Replace with your model file path
#############################################
# Download the model weights
# URL and output file name
url1 = 'https://drive.google.com/uc?id=1Uko0xXO5k0clmRO5F1kOzyiJun69con9'
output1 = 'U_Net_model.h5'

# Attempt to download the file
try:
    gdown.download(url1, output1, quiet=False)
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    print("Download successful!")
except Exception as e:
    print("Error downloading file:", e)

# Assuming you have custom loss and metrics defined
custom_objects = {'jaccard_distance_loss': jaccard_distance_loss, 'dice_coef': dice_coef}

# Load the model
model = load_model('U_Net_model.h5', custom_objects=custom_objects)

######################################### vGG 16
from skimage.io import imread
from skimage.transform import resize

# function to predict result
def predict_image(img_path):
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
# # Load the model with custom objects
# VGG16 = tf.keras.models.load_model('/content/drive/MyDrive/Explo_2024_sem4/VGG16.h5', custom_objects=custom_objects)

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
        mask = make_prediction(image,(224,224,3))
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
