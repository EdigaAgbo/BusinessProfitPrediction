# BusinessProfitPrediction

``` python 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
```
![My Image](dataset.png)

```python 
#Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
en = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder= 'passthrough')
X = np.array(en.fit_transform(X))
print(X)
``
![My Image](dataset.png)
