import pandas as pd
cols_list=["sno","rno","s_name","cgpa","marks","gender"]
df=pd.read_csv("data.csv",usecols=cols_list)
print(df)
mean=df["marks"].mean()
print("MEAN of marks",str(mean))

print("Encoding\n")
from sklearn.preprocessing import LabelEncoder
gencode=LabelEncoder()
df.gender=gencode.fit_transform(df.gender)
print(df)

print("scaling\n")
from sklearn import preprocessing
df.marks=preprocessing.scale(df.marks)
markscale=preprocessing.scale(df.marks)
print(df)

markshape=markscale.reshape(-1,1)
markbin=preprocessing.Binarizer(threshold=0.5).transform(markshape)
df['marks']=markbin
print(df)


duplicate=pd.concat([df]*2,ignore_index=True)
print(duplicate)


duprem=pd.DataFrame.drop_duplicates(duplicate)
print(duprem)

df1=df.copy()
df1['cgpa']=df1['cgpa'].fillna(0)
print(df1)

df2=df.copy()
df2['cgpa'].fillna(df1['cgpa'].mean(),inplace=True)
print(df2)


