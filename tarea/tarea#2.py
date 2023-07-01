#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from zipfile import ZipFile


# In[2]:


import os

os.listdir()


# In[3]:


def unzip_data(path):
    with ZipFile(path,'r') as zipObj:
        zipObj.extractall()


# In[5]:


unzip_data('C:\\Users\\B1\\Downloads\\spaceship-titanic.zip')


# In[17]:


test_ds = pd.read_csv('C:\\Users\\B1\\Documents\\spaceship-titanic_2\\test.csv') 
train_ds = pd.read_csv('C:\\Users\B1\\Documents\\spaceship-titanic_1\\train.csv') 
train_ds.head(5)


# In[18]:


test_ds.head(5)


# In[19]:


ntrain = train_ds.shape[0]  # Calcula la cantidad de filas en el conjunto de entrenamiento
ntest = test_ds.shape[0]  # Calcula la cantidad de filas en el conjunto de prueba

print(f'Dataset has {ntrain} train samples')
print(f'Dataset has {ntest} test samples')


# In[20]:


train_ds.info()


# In[21]:


test_ds.info()


# In[22]:


def impute_most_frequent_data(df):
    for column_name in df.columns:
        data = df[column_name].value_counts().index[0]
        df[column_name].fillna(data, inplace=True)
    return df


# In[24]:


train_ds = impute_most_frequent_data(train_ds)
train_ds.head()


# In[27]:


Nulo= train_ds.isnull().sum()
print(Nulo)


# In[28]:


home_planet_vs_vip = train_ds.groupby('HomePlanet')['VIP'].sum()
home_planet_vs_vip


# In[29]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar(home_planet_vs_vip.index,home_planet_vs_vip)
ax.set_xticklabels(home_planet_vs_vip.index, rotation=45)
ax.set_ylabel("How many of Each Planet are VIP people")
plt.show()


# In[30]:


age_vs_moneyspent =  train_ds.groupby('Age')['RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck'].sum()
age_vs_moneyspent


# In[31]:


fig, ax = plt.subplots(figsize=(10,6))
for i in range(len(age_vs_moneyspent.columns)-2):
    ax.scatter(age_vs_moneyspent.index,age_vs_moneyspent.iloc[:,i], alpha=0.8)
    ax.legend(age_vs_moneyspent.columns)
ax.set_xlabel("Age")
ax.set_xticks(ticks=range(0,80), minor=True)
ax.set_ylabel("Quantity of Money Spent")
ax.grid()

plt.show()


# In[33]:


fig, ax = plt.subplots()
ax.hist(train_ds['Destination'], label="Destination", color='gray')
ax.set_xlabel("Features")
ax.set_ylabel("# of observations")
ax.legend()
plt.show()


# In[34]:


from sklearn.preprocessing import OneHotEncoder


# In[35]:


def column_transform(df, categorical_columns):
    for col in categorical_columns:
        col_ohe = pd.get_dummies(df[col], prefix=col)
        df = pd.concat((df, col_ohe), axis=1).drop(col, axis=1)
    return df


# In[36]:


train_ds_ohe = column_transform(df=train_ds, categorical_columns=["HomePlanet", "Destination"])
train_ds_ohe.head()


# In[37]:


X = train_ds_ohe.drop(['PassengerId', 'Cabin', 'Name', 'Transported'], axis=1)
y = train_ds_ohe['Transported']


# In[58]:


from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# In[64]:


clf = make_pipeline(StandardScaler(),
                    SGDClassifier(max_iter=1000, tol=1e-3))
clf.fit(X, y)


# In[81]:


def preprocess_test_set(test_df):
    test_df = column_transform(df=test_df, categorical_columns=['HomePlanet','Destination'])
    test_df = test_df.drop(['PassengerId', 'Cabin', 'Name'], axis=1)
    return impute_most_frequent_data(test_df)


# In[79]:


# Llamar a la función preprocess_test_set con el test dataset
test_data_processed = preprocess_test_set(test_ds)

# Calcular las predicciones del clasificador con el método predict
y_pred = clf.predict(test_data_processed)

# Transformar las predicciones en un dataframe

pred_df = pd.DataFrame({'Prediction': y_pred})

# Calcular cuántas predicciones fueron pasajeros transportados y cuáles no fueron transportados
prediction_counts = pred_df['Prediction'].value_counts()
print(prediction_counts)


# In[ ]:




