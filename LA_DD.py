#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


#create data
df = pd.DataFrame({'cases': [1743, 1925, 1750, 1447, 1386, 1540, 1519, 1694],
                  'hospitalizations': [396, 357, 304, 288, 322, 345, 371, 422],
                  'deaths': [11, 19, 24, 25, 24, 31, 23, 33]})


# In[5]:


#view data
df


# In[6]:


import statsmodels.api as sm


# In[7]:


#define response variable
y = df['deaths']


# In[8]:


#define predictor variable
x = df[['cases', 'hospitalizations']]


# In[9]:


#add constant to predictor variables
x = sm.add_constant(x)


# In[10]:


#fit linear regression model
model = sm.OLS(y, x).fit()


# In[11]:


#view model summary
print(model.summary())


# In[ ]:




