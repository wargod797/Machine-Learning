
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


home_data = pd.read_csv('train.csv')


# In[3]:


home_data.describe()


# In[4]:


melbourne_data = pd.read_csv('melb_data.csv') 
melbourne_data.columns


# In[5]:


melbourne_data = melbourne_data.dropna(axis=0)


# In[6]:


melbourne_data.Price


# In[7]:


y = melbourne_data.Price


# In[8]:


melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']


# In[9]:


X = melbourne_data[melbourne_features]


# In[10]:


X.describe()


# In[11]:


X.head()


# In[12]:


from sklearn.tree import DecisionTreeRegressor


# In[13]:


melbourne_model = DecisionTreeRegressor(random_state=1)


# In[14]:


#y is the result to expected, X is the data to predict from


# In[15]:


melbourne_model.fit(X, y)


# In[16]:


print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
#underfitting
print(melbourne_model.predict(X.head()))


# In[17]:


from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


# In[20]:


iowa_model = DecisionTreeRegressor(random_state=1)


# In[21]:


iowa_model.fit(train_X, train_y)


# In[22]:


#overfittting model
val_predictions = iowa_model.predict(val_X)


# In[23]:


val_mae = mean_absolute_error(val_predictions, val_y)


# In[24]:


val_predictions


# In[25]:


val_mae


# In[26]:


#Dealing with underfitting and overfitting
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# In[27]:


candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
_
scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
best_tree_size = min(scores, key=scores.get)


# In[28]:


best_tree_size


# In[29]:


# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)

# fit the final model and uncomment the next two lines
final_model.fit(X,y)


# In[30]:


#Random Forest


# In[31]:


from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor()

# fit your model
rf_model.fit(train_X,train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# In[33]:


from sklearn.metrics import mean_absolute_error

melb_preds = rf_val_predictions
print(mean_absolute_error(val_y, melb_preds))

