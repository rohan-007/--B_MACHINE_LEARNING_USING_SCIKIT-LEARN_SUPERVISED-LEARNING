import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import seaborn as sb

data=pd.read_csv('insurance.csv')
data.head()

cols = ['sex', 'smoker', 'region']
new_data = pd.get_dummies(data, cols,drop_first=True)
new_data.head()

new_data.describe()

sb.pairplot(new_data)

sb.heatmap(new_data.corr(),cmap='Blues',annot=True)

rdata=new_data[['age','bmi','smoker_yes','charges']]
rdata.head(10)

ran=np.random.rand(len(data))<0.8
train=rdata[ran]
test=rdata[~ran]
print(train,test,sep='\n')

reg=linear_model.LinearRegression()
train_x=np.asanyarray(train[['age','bmi','smoker_yes']])
train_y=np.asanyarray(train[['charges']])
reg.fit(train_x,train_y)
print(reg.intercept_,reg.coef_)

plt.scatter(train.age,train.charges,color='red')
plt.scatter(train.bmi,train.charges,color='blue')
plt.scatter(train.smoker_yes,train.charges,color='black')

th0=reg.intercept_
th1=reg.coef_[0][0]
th2=reg.coef_[0][1]
th3=reg.coef_[0][2]
a,b,s=[float(i) for i in input("Enter age , bmi , smoker_yes(1/0):").split()]
print(th0+th1*a+th2*b+th3*s)
