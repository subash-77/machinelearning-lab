import matplotlib.pyplot as mtp
from sklearn.cluster import KMeans

x=[4,5,10,4]
y=[24,24,21,21]
data=list(zip(x,y))
#mtp.scatter(x,y)
#mtp.show()

kmeans=KMeans(n_clusters=2,n_init=10)
kmeans.fit(data)

mtp.scatter(x,y,c=kmeans.labels_)
mtp.show()
