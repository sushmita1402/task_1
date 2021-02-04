#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation:Data Science and Business Analytics intern

# # Name-Sushmita chaudhary

# # Task 1:Prediction using Supervised ML

# objective of the tsk:-Predict the percentage of a student based on the number of study hours.

# Dataset link:-http://bit.ly/w-data

# # Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as srb


# In[6]:


df=pd.read_csv('http://bit.ly/w-data')
df


# # Checking whether there are missing values

# In[8]:


df.isna().sum()


# # Viewing first five row by using head() method

# In[10]:


df.head()


# # plotting the data on graph

# In[12]:


plt.scatter(data=df,x="Hours" ,y="Scores")
plt.title("Hours vs percentage")            
plt.xlabel("Hours studied")
plt.ylabel("Scores in percentage")
plt.legend("S")
plt.grid()
plt.show()            
            


# # Data preparation

# In[21]:


x=df.iloc[:,:-1].values
y=df.iloc[:,1].values


# # splitting the data into trainig data and testing data by using train_test_split() method

# In[23]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# # Training the algorithm

# In[24]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)


# # visualising the model

# In[46]:


#plotting the regressor line
line = regressor.coef_*x+regressor.intercept_


# In[49]:


#plotting for the trainig data
plt.scatter(x_train,y_train,color='red')
plt.xlabel('Hours studied')
plt.ylabel('Percentage Score')
plt.grid()
plt.show()


# In[47]:


#plotting for the test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x,line)
plt.xlabel('Hours Studied')
plt.ylabel('percentage Scores')
plt.grid()
plt.show()


# # prediction

# In[31]:


print(x_test)


# In[38]:


y_pred=regressor.predict(x_test)


# In[39]:


#comparing the actual predicted
df=pd.DataFrame({'Actual': y_test,'Predicted': y_pred})
df


# In[43]:


#predicting the score for 9.25 hours
hours=9.25
pred=regressor.predict([[hours]])
print("the predicted score if the student studies for {} hours is {}".format(hours,pred[0]))


# # Evaluating the model

# In[45]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))

