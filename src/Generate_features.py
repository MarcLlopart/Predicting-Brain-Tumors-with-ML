#!/usr/bin/env python
# coding: utf-8

# In[53]:


import joblib
import dill
import pandas as pd
import os
import torchvision
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')


# In[54]:


dataset = joblib.load('dataset.pkl')


# In[55]:


normalize = dill.load(open('normalize.pkl','rb'))
load_dataset = dill.load(open('load_data.pkl','rb'))
transform = joblib.load('image_transform.pkl')
atributs = joblib.load('atributes.pkl')


# In[57]:


#Load dataset
nom_dataset = input("Introdueix el nom del dataset: ")
nom_dataset


# In[58]:


df = load_dataset(nom_dataset)
#dataset_normalitzat = normalize(df)
dataset_net =  df[atributs]


# In[65]:


dataset_net['Class']=dataset['Class']
dataset_net = normalize(dataset_net)
joblib.dump(dataset_net,'dataset_net.pkl')


# In[66]:


nom_imatge = input("Introdueix el nom de la imatge a transformar: ")
joblib.dump(nom_imatge, 'nom_imatge.pkl')


# In[62]:


actual_dir = os.getcwd() 
folder_name="Brain Tumor"
data_dir = os.path.join(actual_dir, folder_name)
image_dir = os.path.join(data_dir, folder_name)
image = os.path.join(image_dir, nom_imatge)
img = Image.open(image)
img_t =  transform(img)


# In[63]:


joblib.dump(img_t, 'imatge_transformada.pkl')


# In[ ]:




