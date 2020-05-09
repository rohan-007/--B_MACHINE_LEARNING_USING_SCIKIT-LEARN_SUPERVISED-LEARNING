import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# TRAINING SET
data=pd.read_csv('FuelConsumption.csv')
data.head()
    
data.describe()

rdata=data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
rdata.head(10)

graph=data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
graph.hist()
plt.show()

ran=np.random.rand(len(data))<0.8
train=rdata[ran]
test=rdata[~ran]
print(train,test,sep='\n')

# ML ALGORITHM
reg=linear_model.LinearRegression()
train_x=np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y=np.asanyarray(train[['CO2EMISSIONS']])
reg.fit(train_x,train_y) 

# PRIDICTIONS
theta0=reg.intercept_[0]
theta1=reg.coef_[0][0]
theta2=reg.coef_[0][1]
theta3=reg.coef_[0][2]
print(theta0,theta1,theta2,theta3)

n=int(input('Number of pridictions:'))
for i in range(n):
    x1,x2,x3=[float(i) for i in input('Enter x1 x2 x3 respectively:').split()]
    print('pridiction is:',(theta0+theta1*x1+theta2*x2+theta3*x3))
