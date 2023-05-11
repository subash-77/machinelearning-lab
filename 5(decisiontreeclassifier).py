import numpy as num
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics

data=pd.read_csv("data3.csv")
features=['age','income']
x=data[features]
y=data.buy

from sklearn.model_selection import train_test_split
xtr,xte,ytr,yte=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier(criterion='entropy',max_depth=3)
regressor.fit(xtr,ytr)

yp=regressor.predict(xte)
xp=regressor.predict(xtr)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y,regressor.predict(x))
print(cm)
print(classification_report(y,regressor.predict(x)))

from sklearn import tree
plt.figure(figsize=(7,7),facecolor='w')
a=tree.plot_tree(regressor,rounded=True,feature_names=features,class_names=y,filled=True,fontsize=12)
plt.show()      
