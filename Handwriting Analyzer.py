#!/usr/bin/env python
# coding: utf-8

# # importing libraries

# In[1]:


import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
import pickle


# In[2]:


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Reshape, BatchNormalization, Input, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *


# In[3]:


df=pd.read_csv("C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Dataset.csv")
df


# train_data="C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training"
# l=len(os.listdir(train_data))
# height=80
# width=2000
# img_data=[]
# for i in os.listdir(train_data):
#     img_list=cv2.imread(os.path.join(train_data,i),cv2.IMREAD_GRAYSCALE)
#     res_img=cv2.resize(img_list,(width,height))
#     cv2.imshow('resized image',res_img)
#     img_data.append((i,res_img))

# # Getting max size of image in training folder

# In[4]:


import cv2
def find_image_with_maximum_size(folder_path):
    max_width = 0
    max_height = 0
    max_image_path = None

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                height, width, _ = image.shape
                if height * width > max_height * max_width:
                    max_width = width
                    max_height = height
                    max_image_path = image_path

    return max_image_path
folder_path = 'C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training/'
max_image_path = find_image_with_maximum_size(folder_path)
if max_image_path is not None:
    print(f"The image with the maximum size is located at: {max_image_path}")
else:
    print("No valid images found in the folder.")


# In[5]:


train_data=cv2.imread("C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training/14.png")
print(train_data.shape)


# # Reading and resize image with max dimension image

# In[6]:


train_data="C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training/"
l=len(os.listdir(train_data))
height=177
width=1886
img_data=[]
for i in os.listdir(train_data):
    image=os.path.join(train_data,i)
    print(image)
    img_list=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('resized image',img_list)
    #cv2.waitKey(0)
    res_img=cv2.resize(img_list,(width,height))
    img_data.append((i,res_img))


# In[7]:


print(img_data)


# In[8]:


def get(na):
    for im_na,im_ar in img_data:
        if na == im_na:
            print("Name matched!",na)
            return im_ar


# In[9]:


df['Name']


# # Create Array col and add image data in it

# In[10]:


df['Array']=df['Name'].apply(get)


# In[11]:


df


# i=cv2.imread("C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training/2.jpg.png")
# cv2.imshow('image',i)
# print(i.shape)
# cv2.waitKey(0)
# 
# 
# 

# import os
# folder_path = 'C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Training'
# file_list = os.listdir(folder_path)
# counter = 1
# for filename in file_list:
#     new_name = str(counter) + os.path.splitext(filename)[1]  
#     old_path = os.path.join(folder_path, filename)
#     new_path = os.path.join(folder_path, new_name)
#     os.rename(old_path, new_path)
#     counter += 1

# In[12]:


plt.imshow(img_data[3][1],cmap='gray')
print(img_data[13][0])


# In[13]:


for k in range(len(img_data)):
    print(img_data[k])


# In[14]:


df['Label_class']=df['Label']


# In[15]:


df['Label_class']


# # Create Label by using encoder 

# In[16]:


encoder = preprocessing.LabelEncoder()


# In[17]:


df['Label_class']=encoder.fit_transform(df['Label_class'])


# In[18]:


Label_classes=encoder.classes_
print(Label_classes)


# In[19]:


print(encoder.classes_,"  ",len(encoder.classes_))


# In[20]:


df


# # Store label in 0 & 1's and storing it as binary file locally

# In[21]:


on=preprocessing.OneHotEncoder()
labels=on.fit_transform(df.Label_class.values.reshape(-1,1)).toarray()


# In[22]:


with open('C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/label_classes','wb') as file:
    pickle.dump(Label_classes,file)
    print('Label successfully stored as binary file for future use!')


# In[23]:


output_classes=len(encoder.classes_)


# In[24]:


print(labels[0])


# # Creating training set by taking array col value 

# In[25]:


training_set=df['Array']
train_set=[]
for img in training_set:
    img=img.reshape(width,height,1)
    train_set.append(img)
train_set=np.array(train_set)


# In[26]:


train_set.shape


# In[27]:


train_labels=df.Label_class.values


# In[34]:


i = Input(shape=(width,height,1))
x = Conv2D(32, (3,3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2,2))(x)
x = Dropout(0.2)(x)

x = GlobalMaxPooling2D()(x)

x = Flatten()(x)

x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(output_classes, activation='softmax')(x)


# In[29]:


model = Model(i,x)


# In[30]:


model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


r = model.fit(train_set,labels, epochs=500,batch_size=42, validation_split=0.2)


# In[31]:


model.save('C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/loadingmodel2.h5')


# In[32]:


img="C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/test1.jpg.png"


# In[33]:


img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array,(width,height))
array = new_array.reshape(-1,width,height, 1)
pred = model.predict(array)
y = np.argmax(pred)
     


# In[ ]:


print(y)
encoder.classes_[y]


# In[ ]:


img="C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/1.jpg.png"
img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
img_array.shape
cv2.imshow("ia",img_array)
cv2.waitKey(0)


# In[ ]:


img_array.shape


# In[ ]:


from keras.models import load_model

# Load the saved model
loaded_model = load_model('C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/loadingmodel2.h5')

img="C:/Users/srinivasa.moorthy/OneDrive - DISYS/Desktop/Testing/test7.png"
img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
new_array = cv2.resize(img_array,(width,height))
array = new_array.reshape(-1,width,height, 1)
pred = loaded_model.predict(array)
y = np.argmax(pred)
     
# Now you can use the loaded_model for making predictions
#predictions = loaded_model.predict(x_test)


# In[ ]:


print(y)
encoder.classes_[y]


# In[ ]:





# In[ ]:




