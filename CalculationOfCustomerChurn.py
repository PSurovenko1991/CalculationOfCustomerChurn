import pandas as pd
import scipy as sp
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from collections import Counter


data_set = pd.read_csv('https://raw.githubusercontent.com/chrisstroud/data/master/Churn_Modelling.csv')
print(data_set.head())



#Formalization:

# geograph = set(data_set['Geography'])
# print(geograph)
# gende = set(data_set['Gender'])
# print(gende)
data_set['Geography'].replace('France',0,inplace= True)  # 
data_set['Geography'].replace('Spain',1,inplace= True)
data_set['Geography'].replace('Germany',2,inplace= True)
data_set['Gender'].replace('Male',0,inplace= True)   
data_set['Gender'].replace('Female',1,inplace= True)

# Correlation matrix:

correlation = data_set.corr()


plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=0.5, square=True,  annot=True, cmap='cubehelix') 
plt.show()


# print(np.sort(correlation.Exited)>0)



print (np.extract(abs((correlation.Exited))>0.11, (correlation.columns)))

# print(corellation.columns)
# print("!!!!!!!!!!!!!!!!!!!!!!!!!!")
# print(correlation.Exited.flags())

# print(correlation.Exited[np.extract(np.sort(correlation.Exited)>0, np.sort(correlation.Exited))])

# print(correlation.Exited[np.extract(np.sort(correlation.Exited)>0, np.sort(correlation.Exited))])

# print(data_set.shape)

train = data_set[:8000]
target = data_set[8001:]

# print(train)
# print("!!!!!!!!!!!!!!!!!!!!")
# print(target)

# for i in data_set.columns:
#     print(i)
#     print(i, (sum(sp.isnan(data_set['Surname']))))
# print(sum(sp.isnan(data_set['Surname'])))/


# X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.25, random_state=0,)

# clf = GaussianNB()
# clf.fit(X_train ,y_train)
# clf.score(X_test, y_test)
