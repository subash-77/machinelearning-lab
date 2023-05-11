import numpy as num
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

data=pd.read_csv("data4.csv")
features=['age','income']
x=data[features]
y=data.buy

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

pca=PCA(n_components=2)
reg=LinearRegression()
pipeline=Pipeline(steps=[('pca',pca),('reg',reg)])
pipeline.fit(x,y)
pred=pipeline.predict(x)

print("Number of feature before PCR",x.shape[1])
print("Number of feature after PCR",pca.n_components_)
print("MAE",metrics.mean_absolute_error(y,pred))
print("MSE",metrics.mean_squared_error(y,pred))
print("RMSE",num.sqrt(metrics.mean_squared_error(y,pred)))
print("R^2",pipeline.score(x,y))


