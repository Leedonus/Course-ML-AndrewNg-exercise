import  pandas as pd
from pandas import DataFrame ,Series
import os
import numpy as np
from pandasgui import  show
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def nomalization(df):
        return df.apply(lambda column :(column-column.mean())/column.std)

def compute_loss(predicts,labels,m):
        loss = 1/(2*m)*np.sum(np.square(predicts-labels))
        return loss 

def grediant_descent(X,theta,labels,m):
        loss = []
        for iter in range(iter_nums):
                predicts = X.dot(theta) 
                update = 1/m *X.T.dot(predicts  -  labels)
                theta =theta - lr*update
                loss.append(compute_loss(predicts,labels,m))
        return theta, loss 

lr = 0.01
iter_nums = 500

root_dir = "/home/leedo/ML_Andrew_Ng/Coursera-ML-AndrewNg-Notes-master/code/ex1-linear regression"
df = pd.read_csv(os.path.join(root_dir,"ex1data1.txt"),names=['population','profit'])

#show(df,settings={'block':True})
#df.plot.scatter(x='df')
#df.plot(x='population',y='profit',kind='scatter')


#df = nomalization(df)
X = np.array(df['population'])
labels = np.array(df['profit'])
labels = np.expand_dims(labels,axis=1)
theta = np.zeros((2,1))

m =df.shape[0]

X= np.concatenate(([[1]]*m,np.expand_dims(X,axis=1)),axis=1)

final_theta , cost_data  = grediant_descent(X,theta,labels,m)

#曲线的斜率和截距
m = final_theta[1,0]
b = final_theta[0,0]


#画出迭代次数和loss曲线
plt.figure(figsize=(6,12))
plt.subplot(211)
plt.title('result')
plt.xlabel("epoches")
plt.ylabel('loss')
plt.plot(np.arange(iter_nums),cost_data)


#画出预测的曲线
plt.subplot(212)
plt.title('result2')
plt.xlabel('population')
plt.ylabel('profit')
plt.scatter(df.population,df.profit,label = 'training_data')
plt.plot(df.population,df.population*m+b,label = 'Prediction',color ='red')
plt.show()






