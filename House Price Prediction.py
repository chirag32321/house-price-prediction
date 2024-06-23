#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction
# 
# In this project, I'll use the California House dataset available in `sklearn` and use `Jupyter Kernel Gateway` to expose its cells as Endpoints.

# ### Import libraries

# In[1]:


import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


# ### Dataset import
# 
# We get the features inside `.data` and labels inside `.target`. We split it into test and train data using `trsin_test_split` with test size of `33%`.

# In[2]:


fetched_data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(fetched_data.data, fetched_data.target, test_size = 0.33)


# Now we will get its desription using `.DESC` and get the complete information on the same.

# In[3]:


print(fetched_data.DESCR)


# ### Data Analysis
# 
# Here, we will analyse the dataset and creata a GET endpoint to fetch the basic stats.

# We first concatenate the features and labels and then combine them as columns with specific column names.

# In[4]:


dataset = pd.concat([pd.DataFrame(fetched_data.data, columns = fetched_data.feature_names), 
                     pd.DataFrame(fetched_data.target*100000, columns = ['Price'])], axis = 1)


# Let's analyse the dataset.

# In[5]:


dataset.info()


# We see that we have a total of 20640 houses. There are total of 8 features and 1 label column. There are no `null` values.

# In[6]:


dataset.corr()


# We see that the price is mainly dependant on Median Income with a correlation of approximately ~0.7.

# #### GET Endpoint
# 
# This endpoint will extract important information about our dataset and return the same when the endpoint is called.

# In[7]:


# GET /housing_stats
total_houses = len(dataset)
max_value = dataset['Price'].describe()['max']
min_value = dataset['Price'].describe()['min']
print(json.dumps({
    'total_houses': total_houses,
    'max_value': max_value,
    'min_value': min_value,
    'most_imp_feature': 'Median Income'
}))


# ### Machine Learning
# 
# Let's now directly train our dataset on the train data and analyse our Mean Absolute Error on the test data.

# In[8]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

clf = RandomForestRegressor(n_estimators = 100, max_depth = 50)
clf.fit(X_train, y_train)
print("Mean Absolute Error: {}".format(mean_squared_error(y_test, clf.predict(X_test))))


# #### POST Endpoint
# 
# Here, I'll train the model on the complete dataset and then simply use the post endpoint to get the price.

# In[9]:


endpoint_classifier = RandomForestRegressor(n_estimators = 100, max_depth = 50)
endpoint_classifier.fit(fetched_data.data, fetched_data.target)


# Let's define a random `REQUEST` object for our POST Endpoint with the mean values from our dataset.

# In[10]:


features = pd.DataFrame(fetched_data.data)
mean_values = features.describe().iloc[1, :]

REQUEST = json.dumps({
    'body': {
        'MedInc': mean_values[0],
        'HouseAge': mean_values[1],
        'AveRooms': mean_values[2],
        'AveBedrms': mean_values[3],
        'Population': mean_values[4],
        'AveOccup': mean_values[5],
        'Latitude': mean_values[6],
        'Longitude': mean_values[7]
    }
})


# The endpoint will accept all the values from the user and return the predicted price. The data received is in the `body` part of the request.

# In[11]:


# POST /get_price
req = json.loads(REQUEST)
req = np.array(list(req['body'].values()))
predicted_price = endpoint_classifier.predict(req.reshape(1, -1))[0]
predicted_price = "{0:.2f}".format(predicted_price*100000)
print(json.dumps({
    'result': 'The price of the house with your specifications should be approximately: $' + predicted_price
}))

