import numpy as num
import matplotlib.pyplot as mtp
import pandas as pd
from sklearn import metrics

data=pd.read_csv("data1.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
print(x)
print("\n")
print(y)

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtr,ytr)

yp=regressor.predict(xte)
xp=regressor.predict(xtr)

print("MAE",metrics.mean_absolute_error(yte,yp))
print("MSE",metrics.mean_squared_error(yte,yp))
print("RMAE",num.sqrt(metrics.mean_absolute_error(yte,yp)))
mtp.scatter(xtr,ytr,color="green")
mtp.plot(xtr,xp,color="red")
mtp.title("OVER VS RUN")
mtp.xlabel("OVER")
mtp.ylabel("RUN")
mtp.show()
