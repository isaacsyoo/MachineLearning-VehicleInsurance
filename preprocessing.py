#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("insurance.csv")
yData = pd.read_csv("yData.csv")
df


# In[3]:


# xTrain1 includes features "region_code" "Policy_Sales_Channel"
df = df.drop(columns=['Response', 'id'])
df.to_csv("data1.csv")
data1 = pd.read_csv("data1.csv", index_col=0)

# # xTrain2 excludes features "region_code" "Policy_Sales_Channel"
ef = df.drop(columns=['Region_Code', 'Policy_Sales_Channel'])
ef.to_csv("data2.csv")
data2 = pd.read_csv("data2.csv", index_col=0)


# In[4]:


# convert "Gender" into binary representation
data1.Gender[data1.Gender == 'Male'] = 0
data1.Gender[data1.Gender == 'Female'] = 1

data2.Gender[data2.Gender == 'Male'] = 0
data2.Gender[data2.Gender == 'Female'] = 1


# In[5]:


# convert "Vehicle_Damage" to binary 
data1.Vehicle_Damage[data1.Vehicle_Damage == 'Yes'] = 0
data1.Vehicle_Damage[data1.Vehicle_Damage == 'No'] = 1

data2.Vehicle_Damage[data2.Vehicle_Damage == 'Yes'] = 0
data2.Vehicle_Damage[data2.Vehicle_Damage == 'No'] = 1


# In[6]:


# convert "Vehicle_Age" into ternary 
data1.Vehicle_Age[data1.Vehicle_Age == '< 1 Year'] = 0
data1.Vehicle_Age[data1.Vehicle_Age == '1-2 Year'] = 1
data1.Vehicle_Age[data1.Vehicle_Age == '> 2 Years'] = 2

data2.Vehicle_Age[data2.Vehicle_Age == '< 1 Year'] = 0
data2.Vehicle_Age[data2.Vehicle_Age == '1-2 Year'] = 1
data2.Vehicle_Age[data2.Vehicle_Age == '> 2 Years'] = 2


# In[7]:


# split data1 into xTrain1, yTrain1, xTest1, yTest1
xTrain1, xTest1, yTrain1, yTest1 = train_test_split(data1, yData, train_size=0.7, test_size=0.3)
xTrain1.to_csv("xTrain1.csv")
xTest1.to_csv("xTest1.csv")
yTrain1.to_csv("yTrain1.csv")
yTest1.to_csv("yTest1.csv")

# split data2 into xTrain2, yTrain2, xTest2, yTest2
xTrain2, xTest2, yTrain2, yTest2 = train_test_split(data2, yData, train_size=0.7, test_size=0.3)
xTrain2.to_csv("xTrain2.csv")
xTest2.to_csv("xTest2.csv")
yTrain2.to_csv("yTrain2.csv")
yTest2.to_csv("yTest2.csv")


# In[33]:


# pearson correlation (heatmap) for data1
coeffs = data1.corr('pearson')
ax = sns.heatmap(coeffs) 


# In[155]:


# pearson correlation (heatmap) for data2
coeffs = data2.corr('pearson')
ax = sns.heatmap(coeffs) 


# In[ ]:


# drop the redundant features
# no need to drop features based on the heatmap 


# In[8]:


xTrain1


# In[9]:


xTrain2


# In[11]:





# In[ ]:




