#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
df=pd.read_csv(r"C:\Users\dell\Downloads\diabetes_data_upload.csv")
print(df)


# In[5]:


X=list(df.Gender.unique())
print(X)


# In[6]:


ColumnHead=list(df.columns)
print(ColumnHead)


# In[7]:


df["Gender"].replace({"Male": 0,"Female": 1},inplace=True)
print(df)


# In[8]:


ColumnHead1=ColumnHead[2:-1]
print(ColumnHead1)


# In[15]:


for i in ColumnHead:
    J=list(df[i].unique())
    print(J)


# In[16]:


for k in ColumnHead1:
    df[k].replace({"Yes": 0, "No": 1},inplace=True)
print(df)


# In[20]:


df["class"].replace({"Positive": 0, "Negative": 1},inplace=True)
print(df)


# In[13]:


for ind in df.index:
    if df["Age"][ind]<=30:
        df["Age"][ind]=0
    elif df["Age"][ind]>30 and df["Age"][ind]<=60:
        df["Age"][ind]=1
    else:
        df["Age"][ind]=2
    print(df["Age"][ind])
        


# In[ ]:





# In[21]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X = df.drop('class',axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

model = LogisticRegression(random_state=10)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))


# In[ ]:





# In[ ]:




