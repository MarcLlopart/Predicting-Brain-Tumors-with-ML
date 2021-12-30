#!/usr/bin/env python
# coding: utf-8

# In[4]:


import joblib
import dill
import pandas as pd
import os
import torchvision
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[10]:


dataset_net = joblib.load('dataset_net.pkl')
atributs = joblib.load('atributes.pkl')
model_escollit=joblib.load('model_escollit.pkl')
y = dataset_net.iloc[:,-1]
X = dataset_net.loc[:,atributs]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = joblib.load('trained_model.pkl')
print(f"El model seleccionat és {model_escollit} i té una accuracy del {round(model.score(X_test, y_test),3)*100}%")


# In[7]:


out = joblib.load('trained_images.pkl')
labels = joblib.load('labels.pkl')
nom_imatge = joblib.load('nom_imatge.pkl')

_, index = torch.max(out, 1)
 
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

if (labels[index[0]] == "Tumor"):
    pacient = "tenir un tumor cerebral"
else: 
    pacient = "no tenir un tumor cerebral"
print(f"El pacient de la {nom_imatge} té {round(percentage[index[0]].item(),2)}% de possiblitats de {pacient}")

