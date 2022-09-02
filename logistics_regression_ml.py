#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("insurance_data.csv")
df.head()


# In[4]:


plt.scatter(df.age, df.bought_insurance, marker='+', color='red')


# In[6]:


df.shape


# In[7]:


from sklearn.model_selection import train_test_split


# In[24]:


train_test_split(df[['age']], df.bought_insurance, test_size=0.1)


# In[11]:


X_test


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[16]:


model =  LogisticRegression()


# In[17]:


model.fit(X_train, y_train)


# In[18]:


model.predict(X_test)


# In[19]:


model.score(X_test,y_test)


# In[20]:


model.predict_proba(X_test)


# In[ ]:





# In[ ]:




