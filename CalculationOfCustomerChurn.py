import pandas as pd
import scipy as sp
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB


data_set = pd.read_csv('https://raw.githubusercontent.com/chrisstroud/data/master/Churn_Modelling.csv') #or attached file

print(data_set.head())

#Formalization:

for i in set(data_set["Geography"]):
    data_set.insert(data_set.columns.size-1,i,data_set['Geography']== i)
data_set = data_set.drop(["Geography"], axis=1)

data_set['Gender'].replace('Male',0,inplace= True)   
data_set['Gender'].replace('Female',1,inplace= True)



#Correlation matrix:

correlation = data_set.corr()


plt.figure(figsize=(25,25))
sns.heatmap(correlation, vmax=1, square=True,  annot=True, cmap='cubehelix') 
plt.show()

#Find important fields:
fields =  (np.extract(abs((correlation.Exited))>0.12, (correlation.columns)))
fields = fields[:-1]
print("Important fields: ",fields)



#Formation of test sets:

train = data_set[fields]
target = data_set["Exited"]

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.6, random_state=0,)

#Training:
clf = GaussianNB()
clf.fit(X_train ,y_train)

#Test:
clf.score(X_test, y_test)



