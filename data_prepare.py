import numpy as np
import os
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
from scipy import ndimage as ndi
from torchvision import datasets, models, transforms

import tensorflow as tf
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
import random
from skimage import io, transform, color, exposure
from skimage import data, img_as_float, img_as_ubyte,morphology
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Set the data path
data_dir  ='Blood cell Cancer [ALL]'
data_list = sorted(list(paths.list_images(data_dir)))

# Seperate the training and test data
random.seed(88)
random.shuffle(data_list)
train_list, test_list = train_test_split(data_list, train_size=0.90, shuffle=True, random_state=88)

print('number of testing list -:',len(test_list))
print('number of training list-:',len(train_list))

#Data overview
print('Number of samples in dataset:',len(list(paths.list_images("Blood cell Cancer [ALL]"))))

print('Number of samples in each class:','\n')
print("#1 Benign ---------------:", len(list(paths.list_images("Blood cell Cancer [ALL]/Benign"))))
print("#2 Malignant[Early PreB] :", len(list(paths.list_images("Blood cell Cancer [ALL]/[Malignant] early Pre-B"))))
print("#3 Malignant[PreB] ------:", len(list(paths.list_images("Blood cell Cancer [ALL]/[Malignant] Pre-B"))))
print("#4 Malignant[ProB] ------:", len(list(paths.list_images("Blood cell Cancer [ALL]/[Malignant] Pro-B"))))


#Process the data
p=0
for img in test_list[:]:
  i=cv.imread(img)
  i=cv.resize(i,(224,224))
  label=img.split(os.path.sep)[1]
  if (label=="Benign"):
    b= ('data_ready/test_data/Benign/Benign'+str(p)+'.png')
  if (label=="[Malignant] Pre-B"):
    b= ('data_ready/test_data/Malignant Pre-B/Malignant Pre-B'+str(p)+'.png')
  if (label=="[Malignant] Pro-B"):
    b= ('data_ready/test_data/Malignant Pro-B/Malignant Pro-B'+str(p)+'.png')
  if (label=="[Malignant] early Pre-B"):
    b= ('data_ready/test_data/Malignant early Pre-B/Malignant early Pre-B'+str(p)+'.png')
  p=p+1
  cv.imwrite(b,i)

p=0

for img in train_list[:]:

    i= cv.imread(img)
    i= cv.resize(i,(224,224))
    label= img.split(os.path.sep)[2]
    if (label=="Benign"):
      b= ('data_ready/test_data/Benign/Benign'+str(p)+'.png')
    if (label=="[Malignant] Pre-B"):
      b= ('data_ready/test_data/Malignant Pre-B/Malignant Pre-B'+str(p)+'.png')
    if (label=="[Malignant] Pro-B"):
      b= ('data_ready/test_data/Malignant Pro-B/Malignant Pro-B'+str(p)+'.png')
    if (label=="[Malignant] early Pre-B"):
      b= ('data_ready/test_data/Malignant early Pre-B/Malignant early Pre-B'+str(p)+'.png')
    p+=1
    cv.imwrite(b,i)
    i= cv.cvtColor(i, cv.COLOR_BGR2RGB)
    i_lab = cv.cvtColor(i, cv.COLOR_RGB2LAB)        #RGB -> LAB
    l,a,b = cv.split(i_lab)
    i2 = a.reshape(a.shape[0]*a.shape[1],1)
    km= KMeans(n_clusters=7, random_state=0,n_init=10).fit(i2)  #Clustring
    p2s= km.cluster_centers_[km.labels_]
    ic= p2s.reshape(a.shape[0],a.shape[1])
    ic = ic.astype(np.uint8)
    r,t = cv.threshold(ic,141,255 ,cv.THRESH_BINARY) #Binary Thresholding
    fh = ndi.binary_fill_holes(t)                      #fill holes
    m1 = morphology.remove_small_objects(fh, 200)
    m2 = morphology.remove_small_holes(m1,250)
    m2 = m2.astype(np.uint8)
    out = cv.bitwise_and(i, i, mask=m2)
    if (label=="Benign"):
      b= ('data_ready/test_data/Benign/Benign'+str(p)+'.png')
    if (label=="[Malignant] Pre-B"):
      b= ('data_ready/test_data/Malignant Pre-B/Malignant Pre-B'+str(p)+'.png')
    if (label=="[Malignant] Pro-B"):
      b= ('data_ready/test_data/Malignant Pro-B/Malignant Pro-B'+str(p)+'.png')
    if (label=="[Malignant] early Pre-B"):
      b= ('data_ready/test_data/Malignant early Pre-B/Malignant early Pre-B'+str(p)+'.png')
    p+=1
    out= cv.cvtColor(out, cv.COLOR_RGB2BGR)
    cv.imwrite(b,out)