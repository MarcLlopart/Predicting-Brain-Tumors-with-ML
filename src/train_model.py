#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# Escull entre els models següents el que més t'agradi: 
# 
# 1. Regressió Logística  
# 2. Decision Tree
# 3. Random Forest
# 4. AdaBoostClassifier
# 5. XGBoostClassifier
# 6. MLP (Neuronal Network)
# 7. SVM Kernel Linear
# 8. LinearSVC
# 9. SVM Kernel RBF
# 10. SVM Kernel Polynomial
# 11. SVM Kernel Sigmoid

# In[29]:


print("1. Regressió Logística\n2. Decision Tree\n3. Random Forest\n4. AdaBoostClassifier\n5. XGBoostClassifier\n6. MLP (Neuronal Network)\n7. SVM Kernel Linear\n8. LinearSVC\n9. SVM Kernel RBF\n10. SVM Kernel Polynomial\n11. SVM Kernel Sigmoid ")
model_escollit = int(input("Tria el teu model: "))


# In[28]:


if model_escollit ==1 : 
    model_escollit="Regressió Logística"
    model = joblib.load('lr.pkl') # Guardo el modelo.
elif model_escollit ==2:
    model_escollit="Decision Tree"
    model = joblib.load('dt.pkl') # Guardo el modelo.
elif model_escollit ==3: 
    model_escollit="Random Forest"
    model = joblib.load('rf.pkl')
elif model_escollit ==4: 
    model_escollit="Ada Boost Classifier"
    model = joblib.load('ada.pkl') # Guardo el modelo.
elif model_escollit ==5: 
    model_escollit="XGBoost Classifier"
    model = joblib.load('xb.pkl') # Guardo el modelo.
elif model_escollit ==6:
    model_escollit="MLP(Neuronal Network)"
    model = joblib.load('mlp.pkl') # Guardo el modelo.
elif model_escollit ==7:
    model_escollit="SVM Kernel Linear"
    model = joblib.load('sl.pkl') # Guardo el modelo.
elif model_escollit ==8:
    model_escollit="LinearSVC"
    model = joblib.load('svcl.pkl') # Guardo el modelo.
elif model_escollit ==9:
    model_escollit="SVM Kernel RBF"
    mnodel = joblib.load('rbf.pkl') # Guardo el modelo.
elif model_escollit ==10:
    model_escollit="SVM Kernel Polynomial"
    model = joblib.load('polynomical.pkl') # Guardo el modelo.
elif model_escollit ==11:
    model_escollit="SVM Kernel Sigmoid"
    model = joblib.load('sigmoid.pkl') # Guardo el modelo.

squeezenet = joblib.load('squeezenet.pkl') # Guardo el modelo.
joblib.dump(model_escollit,'model_escollit.pkl')


# In[23]:


dataset_net = joblib.load('dataset_net.pkl')
atributs = joblib.load('atributes.pkl')


# In[24]:


y = dataset_net.iloc[:,-1]
X = dataset_net.loc[:,atributs]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
model = model.best_estimator_
model = model.fit(X_train, y_train)


# In[25]:


joblib.dump(model, 'trained_model.pkl')


# In[27]:


img_t = joblib.load('imatge_transformada.pkl')
labels = ["No tumor", "Tumor"]
batch_t = torch.unsqueeze(img_t,0)
img_model = squeezenet.eval()
out = img_model(batch_t)
joblib.dump(out,'trained_images.pkl')
joblib.dump(labels,'labels.pkl')


# In[ ]:




