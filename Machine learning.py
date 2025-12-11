import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
#Three types of machine learning problems - classification(sort into categories), rergression(predict the next value), recommendation(what should be recommended based on past data)

#Linear regression
x = np.arange(1,11)
y = []
for i in range(10):
    r = random.randint(20,30)
    y.append(r)
xmean = sum(x)/len(x)
font1 = {"family":"serif","color":"blue","size":10}
plt.title = ('temperature/days graph',font1)
plt.xlabel("Days")
plt.ylabel("Temperature")
plt.plot(x,y,linewidth = 5,color = "Red",linestyle = "dotted",marker = "x",label = "line 1")
print(xmean)
num = 0
den = 0
ymean = sum(y)/len(y)
#m = sum((xi-xmean)*(yi-ymean))/sum((xi-xmean)**2)
for i in range(len(x)):
    num = num + (x[i]-xmean)*(y[i]-ymean)
    den = den + (x[i]-xmean)**2
m = num/den
c = ymean-m*xmean
print("slope is:",m,"y-intercept is:",c)
#Prediction of y
predy = m*x+c
plt.plot(x,predy,linewidth = 5, color = 'Blue', linestyle = "solid", label = "line 2")
plt.show()

#Evaluating model
#RMSE stands for root mean square error( do in reverse order)
error = np.sqrt(np.mean((predy-y)**2))
print(error)

#Applying linear regression using library class
from sklearn.linear_model import LinearRegression

r1 = LinearRegression()
x = x.reshape(-1,1)
r1.fit(x,y)
print(r1.coef_)
print(r1.intercept_)
y2y = r1.predict(x)
print(y2y)
from sklearn.metrics import root_mean_squared_error
err = root_mean_squared_error(y2y,y)
print(err)