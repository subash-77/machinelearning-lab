import matplotlib.pyplot as mtp
from scipy.cluster.hierarchy import dendrogram,linkage

x=[4,5,10,4]
y=[24,24,21,21]
data=list(zip(x,y))
mtp.scatter(x,y)
mtp.show()

linkage_data=linkage(data,method='ward',metric='euclidean')
dendrogram(linkage_data)
mtp.show()
