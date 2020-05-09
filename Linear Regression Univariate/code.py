# importing some necessary modules for MACHINE LEARNING
import pandas as pd               # Dealing with input training set data
import matplotlib.pyplot as plt   # Dealing with ploting the training set and output  (pyplot is class in matplotlib module)
import numpy as np                # Converting data set to matrix form
from sklearn import linear_model  # Helps to find parameters of hypothesis { MACHINE LEARNING ALGORITHM }

# Training Set
data=pd.read_csv('house_prices_updated.csv')  # Creating data object for read_csv() class { Stores data from file to data variable }
data.head()                                   # object.method()  { To display data present in data object }

# TRAINING SET
rdata=data[['SIZE','PRICE']]    # Selecting particular features from data object to rdata { new object}
rdata.head()                    # Display data present in rdata object

# TRAINING SET
graph=rdata[['SIZE','PRICE']]    # Selecting variables for ploting
graph.hist()                     # object.method()  for ploting histogram graph
plt.show()                       # Show the created graph

# TRAINING SET
plt.scatter(rdata.SIZE,rdata.PRICE,color='red')   # Selecting variables for scatter
plt.xlabel('SIZE ----->')                         # labeling x axis
plt.ylabel('PRICE ---->')                         # labeling y axis
plt.show()                                        # show the created graph

# TRAINING SET -- SELCTING 80% TARINING SET AND 20% TEST SET
ran=np.random.rand(len(data))<0.8
train=rdata[ran]
test=rdata[~ran]
print(train,test,sep='\n')

# Algorithm implementation using sk learn
reg=linear_model.LinearRegression()        # Object reg for class LinearRegression
train_x=np.asanyarray(train[['SIZE']])     # Matrix form
train_y=np.asanyarray(train[['PRICE']])    # Matrix form
reg.fit(train_x,train_y)
print(reg.coef_,reg.intercept_,sep='  ')            # Printing theta 0 { intercept}  and theta 1 { slope} i.e parameters of hypothesis

# ploting output
plt.scatter(train.SIZE, train.PRICE,  color='blue')                         # Input plot
plt.plot(train.SIZE, reg.coef_[0][0]*train_x + reg.intercept_[0], '-r')     # Output plot
plt.xlabel('SIZE')
plt.ylabel('PRICE')
plt.show()
   
    
    
