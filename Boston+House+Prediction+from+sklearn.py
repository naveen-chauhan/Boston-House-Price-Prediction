
# coding: utf-8

# In[1]:


from sklearn.datasets import load_boston


# In[2]:


boston=load_boston()


# In[3]:


boston.keys()


# In[4]:


boston.DESCR


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[6]:


data=pd.DataFrame(boston.data)


# In[7]:


data.head()


# In[8]:


data.columns=boston.feature_names


# In[9]:


data.head()


# In[10]:


target=pd.DataFrame(boston.target)
target.columns=['MEDV']


# In[11]:


len(target.loc[target.MEDV==50.0,])


# In[12]:


data['MEDV']=target['MEDV']


# In[13]:


data.head()


# In[14]:


ax=plt.subplots(figsize=(15,9))
import seaborn as sns
correl=data.corr()
sns.heatmap(correl,vmax=0.8,square=True)


# In[15]:


from sklearn.cross_validation import ShuffleSplit


# In[16]:


price=target.MEDV


# In[17]:


features=data.loc[:,['RM','PTRATIO','LSTAT']]


# In[18]:


features.head()


# In[19]:


minimum_price=np.min(price)


# In[20]:


maximum_price=np.max(price)


# In[21]:


mean_price=np.mean(price)


# In[22]:


median_price=np.median(price)


# In[23]:


std_price=np.std(price)


# In[24]:


first_quartile=np.percentile(price,25)
third_quartile=np.percentile(price,75)
inter_quartile=third_quartile-first_quartile


# In[25]:


print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print ("Maximum price: ${:,.2f}".format(maximum_price))
print ("Mean price: ${:,.2f}".format(mean_price))
print ("Median price ${:,.2f}".format(median_price))
print ("Standard deviation of prices: ${:,.2f}".format(std_price))
print ("First quartile of prices: ${:,.2f}".format(first_quartile))
print ("Second quartile of prices: ${:,.2f}".format(third_quartile))
print ("Interquartile (IQR) of prices: ${:,.2f}".format(inter_quartile))


# In[26]:


for i,col in enumerate(features.columns):
    plt.subplot(1,3,i+1)
    x=data[col]
    y=price
    plt.plot(x,y,'o')
    plt.plot(np.unique(x),np.poly1d(np.polyfit(x,y,1))(np.unique(x)))
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('price')


# In[44]:


from sklearn.metrics import r2_score
def performance_metric(y_true,y_pred):
    accuracy_score=r2_score(y_pred,y_true)
    print("the accuracy of classification  ",accuracy_score)
    return accuracy_score 


# In[28]:


#now split the dataset
from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y=train_test_split(features,price,random_state=42)


# In[29]:


train_X.shape


# In[48]:


from sklearn.metrics import make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def fit_model(X,y):
    cv_sets=ShuffleSplit(X.shape[0],n_iter=10,test_size=.20,random_state=0)
    regressor=DecisionTreeRegressor()
    count=range(1,11)
    params=dict(max_depth=count)
    #now grid Search cv
    scoring_func=make_scorer(performance_metric)
    grid=GridSearchCV(regressor,params,cv=cv_sets,scoring=scoring_func)
    grid=grid.fit(X,y)
    return grid.best_estimator_


# In[55]:


from sklearn.model_selection import RandomizedSearchCV
def fit_model2(X,y):
    cv_sets=ShuffleSplit(X.shape[0],n_iter=10,test_size=.20,random_state=0)
    regressor=DecisionTreeRegressor()
    count=range(1,11)
    params=dict(max_depth=count)
    scoring_func=make_scorer(performance_metric)
    grid=RandomizedSearchCV(regressor,params,cv=cv_sets,scoring=scoring_func)
    grid=grid.fit(X,y)
    return grid.best_estimator_


# In[56]:


reg=fit_model(train_X,train_y)
reg.get_params()['max_depth']


# In[57]:


reg_2=fit_model2(train_X,train_y)
reg_2.get_params()


# In[58]:


client_data=[[5,17,15],
             [4,32,22],
             [8,3,12]]


# In[59]:


print(reg.predict(client_data))


# In[62]:


price.hist(bins=50)
for prices in reg.predict(client_data):
    plt.axvline(prices,lw=5,c='r')

