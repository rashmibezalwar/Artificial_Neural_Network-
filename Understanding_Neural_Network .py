#!/usr/bin/env python
# coding: utf-8

# # Artificial Neural Network

# ### Importing the libraries

# In[4]:


pip install tensorflow


# In[6]:


import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report,confusion_matrix
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


tf.__version__


# ## Part 1 - Data Preprocessing

# ### Importing the dataset

# In[9]:


dataset = pd.read_csv("C:/Users/Rashmi/Desktop/AI Data_Excelr/Neural Network/Churn_Modelling.csv")


# In[10]:


dataset.head()


# In[11]:


X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values


# In[12]:


print(X)


# In[13]:


print(y)


# ### Encoding categorical data

# Label Encoding the "Gender" column

# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])


# In[15]:


print(X)


# ### Splitting the dataset into the Training set and Test set

# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[17]:


X_train.shape , y_train.shape


# In[18]:


X_test.shape , y_test.shape


# ### Feature Scaling

# In[14]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[15]:


X_train


# ## Part 2 - Building the ANN

# ### Initializing the ANN

# In[16]:


ann = tf.keras.models.Sequential()


# ### Adding the input layer

# In[17]:


ann.add(tf.keras.layers.Dense(units=9, activation='relu'))


# ### Adding the hidden layers

# In[18]:


ann.add(tf.keras.layers.Dense(units=16, activation='relu'))  ### hidden layer 1

ann.add(tf.keras.layers.Dense(units=16, activation='relu'))  ### hidden layer 2

ann.add(tf.keras.layers.Dense(units=8, activation='relu'))   ### hidden layer 3


# ### Adding the output layer

# In[19]:


ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))


# ## Part 3 - Training the ANN

# ### Compiling the ANN

# In[20]:


ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# ### Training the ANN on the Training set

# In[21]:


ann.fit(X_train, y_train, batch_size = 100, epochs = 50)


# In[22]:


ann.summary()


# In[23]:


y_pred_tr = ann.predict(X_train)


# In[24]:


y_pred_tr


# In[25]:


y_pred_tr = (y_pred_tr > 0.5)
print(np.concatenate((y_pred_tr.reshape(len(y_pred_tr),1), y_train.reshape(len(y_train),1)),1))


# In[26]:


import sklearn.metrics as metrics
print("Accuracy :", metrics.roc_auc_score(y_train, y_pred_tr))


# In[41]:





# In[42]:


from sklearn.metrics import confusion_matrix

log_cm = confusion_matrix(y_train, y_pred_tr.reshape(len(y_pred_tr),1))

sns.heatmap(log_cm, annot=True, fmt='.2f',
         xticklabels = ["0", "1"] , yticklabels = ["0", "1"] )

plt.ylabel('Actual')
plt.xlabel('Predicted')


# ### Predicting the Test set results

# In[33]:


y_pred_tst = ann.predict(X_test)
y_pred_tst = (y_pred_tst > 0.5)
print(np.concatenate((y_pred_tst.reshape(len(y_pred_tst),1), y_test.reshape(len(y_test),1)),1))


# In[34]:


print("Accuracy :", metrics.roc_auc_score(y_test, y_pred_tst))


# In[44]:


from sklearn.metrics import confusion_matrix

log_cm = confusion_matrix(y_test, y_pred_tst.reshape(len(y_pred_tst),1))

sns.heatmap(log_cm, annot=True, fmt='.2f',
         xticklabels = ["0", "1"] , yticklabels = ["0", "1"] )

plt.ylabel('Actual')
plt.xlabel('Predicted')


# In[ ]:





# ## Part 4 - Making the predictions and evaluating the model

# ### Predicting the result of a single observation

# **Homework**
# 
# Use our ANN model to predict if the customer with the following informations will leave the bank or not: 
# 
# 
# Credit Score: 600
# 
# Gender: Male
# 
# Age: 40 years old
# 
# Tenure: 3 years
# 
# Balance: \$ 60000
# 
# Number of Products: 2
# 
# Does this customer have a credit card ? Yes
# 
# Is this customer an Active Member: Yes
# 
# Estimated Salary: \$ 50000
# 
# So, should we say goodbye to that customer ?

# **Solution**

# In[58]:


print(ann.predict(sc.transform([[600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# In[ ]:





# In[ ]:





# ### Save the Model

# In[45]:


### import load_mode
from keras.models import load_model 

### save the model as .h5 file
ann.save("ann_clf.h5") 

### load the saved model
ann_model = load_model("ann_clf.h5") 


# In[46]:


loss, accuracy = ann_model.evaluate(X_test, y_test) 


# In[ ]:





# In[47]:


### prediction for future data

print(ann_model.predict(sc.transform([[600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)


# In[ ]:





# In[ ]:




