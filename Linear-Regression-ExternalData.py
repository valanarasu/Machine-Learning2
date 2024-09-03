from sklearn.datasets import load_diabetes
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=load_diabetes()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
dataset['Price']=df.target
dataset.head()
x=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]
x.head()
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)
# from scipy import stats

from sklearn.model_selection import cross_val_score
mse=cross_val_score(lin_reg,x,y,scoring='neg_mean_absolute_error',cv=5)
mean_mse=np.mean(mse)
print(mean_mse)
# print(lin_reg.predict([2],[2]))
