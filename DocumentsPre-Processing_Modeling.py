#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from  sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import  GridSearchCV
get_ipython().run_line_magic('matplotlib', 'inline')
pd.pandas.set_option('display.max_columns',None)


# In[36]:


os.chdir('/Users/kofiassabil/Downloads')


# In[37]:


df = pd.read_csv('dataWrangling.csv')
df.head()


# In[38]:


df = df.drop(['GarageYrBlt'], axis =1)


# In[39]:


df.shape


# In[40]:


df.columns


# In[41]:


#feature Engineering
df = df.drop(['Unnamed: 0','Id'], axis=1)


# In[42]:


for feature in ['YearBuilt','YearRemodAdd']: 
    df[feature]=df['YrSold']-df[feature]


# In[43]:


numfeatures = ['LotArea','1stFlrSF','GrLivArea','LotFrontage','SalePrice']
for feature in numfeatures:
    df[feature] = np.log(df[feature])

df.head()


# In[44]:


# select highly correlated features
# and remove the first feature that is correlated with anything other feature

def cor(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr
corr_features = cor(df, 0.7)
len(set(corr_features))


# In[45]:


#the highly correlated features
corr_features


# In[46]:


#drop the first and last one
df.drop(['1stFlrSF','TotRmsAbvGrd'],axis=1)


# In[47]:


dummy = pd.get_dummies(df)
dummy.shape


# In[48]:


finalSet = dummy.drop(['SalePrice'], axis=1)
finalSet.head()


# In[49]:


#split into trainging and testing set
X=finalSet
y=df['SalePrice']
scaler = preprocessing.StandardScaler().fit(X)
X_scaled=scaler.transform(X)
X_scaled


# In[50]:


y = y.ravel()
X_train, X_test, y_train,y_test= train_test_split(X_scaled, y, test_size=0.25, random_state=1)
X_train.shape, X_test.shape


# In[51]:


scale=[feature for feature in finalSet.columns ]


# In[52]:


# transform the sets and add the Id and SalePrice variables
data = pd.concat([df[['SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(finalSet[scale]), columns=scale)],
                    axis=1)


# In[53]:


data.isna().any()[lambda x: x]


# In[54]:


data


# In[55]:


data.to_csv('x_train.csv',index=False)


# In[56]:


df = pd.read_csv('x_train.csv')


# In[57]:


df.head()


# In[58]:


#DV
y = df[['SalePrice']]
#drop DV from dataset and IV
x = df.drop(['SalePrice'],axis=1)


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)


# In[60]:



#model 1
lr = LinearRegression()
model = lr.fit(X_train, y_train)


# In[61]:


predictions = model.predict(X_test)


# In[62]:


print("R^2 is: ", model.score(X_test, y_test))
print("RMSE is: ", mean_squared_error(y_test, predictions))


# In[63]:


#model 2
#lasso 
lasso = Lasso()
params = {"alpha" : [ 1e-3, 1e-2, 1, 1e1, 
                     1e2, 1e3, 1e4, 1e5, 1e6, 1e7]}


# In[65]:


lasso_regress = GridSearchCV(lasso, params, cv=5)
lasso_model= lasso_regress.fit(X_train, y_train)
lasso_pred = lasso_model.predict(X_test)


# In[66]:


print("R^2 is: ", lasso_model.score(X_test, y_test))
print("RMSE is: ", mean_squared_error(y_test, lasso_pred))


# In[67]:


lasso_regress.best_estimator_


# In[ ]:




