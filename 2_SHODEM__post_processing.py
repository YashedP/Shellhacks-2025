# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 16:43:36 2023

@code author: Pietro Scala, University of Palermo 

Cite as. Scala et al., 2023

"""

from tensorflow.keras import backend as K
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
import cv2 as cv


directory = os.getcwd()
print(directory)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection +1.0)

model = keras.models.load_model('C:\\Users\\Utente\\Desktop\\AUGMENTATION\\SHODEM_v1.1',custom_objects={'jacard_coef':jacard_coef},compile=False)
history1=np.load('C:\\Users\\Utente\\Desktop\\AUGMENTATION\\SHODEM_v1.1\\SHODEM_history.npy',allow_pickle='TRUE').item()

model.summary()
#path_attuale=QgsProject.instance().homePath();
#os.chdir(path_attuale)


history = history1
loss = history['loss']
val_loss = history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training and validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

history = history1
accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'y', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

acc = history['jacard_coef']
val_acc = history['val_jacard_coef']

plt.plot(epochs, acc, 'y', label='Training IoU')
plt.plot(epochs, val_acc, 'r', label='Validation IoU')
plt.title('Training and validation IoU')
plt.xlabel('Epochs')
plt.ylabel('IoU')
plt.legend()
plt.show()

# Resize process below if the image doesn't have the correct size for the model run

basewidth = 512
img = Image.open('C:\\Users\\Utente\\Desktop\\OUTPUT\\IMMAGINI_ATA_2013\\JPG_S.jpg')
wpercent = (basewidth / float(img.size[0]))
hsize = 512
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img.save('C:\\Users\\Utente\\Desktop\\OUTPUT\\JPG_r.png')
im = Image.open('C:\\Users\\Utente\\Desktop\\OUTPUT\\JPG_r.png')
rgb_im = im.convert('RGB')
rgb_im.save('C:\\Users\\Utente\\Desktop\\OUTPUT\\JPG_res.jpg')

test_img = cv2.imread('C:\\Users\\Utente\\Desktop\\OUTPUT\\JPG_res.jpg')

#test_img.shape
test_img_input=np.expand_dims(test_img, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
data=predicted_img

plt.figure()
imgplot = plt.imshow(test_img)
plt.show()

plt.figure()
imgplot = plt.imshow(predicted_img)
plt.show()


# IMPORTANT: Save the predicted (SEGMENTED) image as cv

imm = data
imm = cv2.imread('Load the predicted image')

indices = imm<140

indices = indices.astype(np.uint8)  #convert to an unsigned byte
indices*=255

cv2.waitKey()
am = Image.fromarray(indices)
am.save('C:\\Users\\Utente\\Desktop\\OUTPUT\\IMMAGINI_ATA_2013\\JPG_true.jpg')


img = img = cv.imread('C:\\Users\\Utente\\Desktop\\OUTPUT\\IMMAGINI_ATA_2013\\JPG_true.jpg',0)
img = cv2.medianBlur(img,5)

#cv.AdaptiveThreshold(src, dst, maxValue, adaptive_method=CV_ADAPTIVE_THRESH_MEAN_C, thresholdType=CV_THRESH_BINARY, blockSize=3, param1=5) 

ret,th1 = cv2.threshold(img,20,20,cv2.THRESH_BINARY)


titles = ['Original Image', 'Final Image']         

Twoplots= [test_img, th1]

for i in range(2):
    plt.subplot(2,2,i+1),plt.imshow(Fourplots[i])
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

