import numpy as num
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

data=pd.read_csv("data2.csv")
x=data.iloc[:,:-1].values
y=data.iloc[:,1].values
print(x)
print("\n")
print(y)

from sklearn.linear_model import LogisticRegression
regressor=LogisticRegression(solver='liblinear',random_state=0)
regressor.fit(x,y)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y,regressor.predict(x))
print(classification_report(y,regressor.predict(x)))
print(cm)

fig,ax=plt.subplots(figsize=(8,8))
ax.imshow(cm)
ax.xaxis.set(ticks=(0,1),ticklabels=('predicted 0s','predicted 1s'))
ax.yaxis.set(ticks=(0,1),ticklabels=('actual 0s','actual 1s'))
ax.set_ylim(1.5,-0.5)
for i in range(2):
    for j in range(2):
        ax.text(j,i,cm[i,j],ha='center',va='center',color="red")
plt.show()
