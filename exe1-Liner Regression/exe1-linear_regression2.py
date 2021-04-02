import os
import numpy as np
from matplotlib import pyplot  as plt

import pandas as pd
from pandasgui import  show
from pandas import DataFrame ,Series


def normalization(df):
        return df.apply(lambda x :(x-x.mean())/x.std())


def  compute_loss(predicts,labels,m):
        return 1/(2*m)*np.sum(np.square(predicts-labels))

def grediant_decent(alpha,m,X,theta,labels):
        loss=[]
        for i in range(iter_nums):
                predicts = X.dot(theta)
                update =  (1/m)*X.T.dot(predicts-labels)
                theta =theta - alpha*update
                loss.append(compute_loss(predicts,labels,m))
        return theta,loss

#读取数据
root_dir = "/home/leedo/ML_Andrew_Ng/Coursera-ML-AndrewNg-Notes-master/code/ex1-linear regression"
df = pd.read_csv(os.path.join(root_dir,"ex1data2.txt"),names=["param1","param2","param3"])
#show(df,settings={'block':True})

#学习率和迭代次数
alpha_candidate= [0.001,0.0033,0.01,0.033,0.1]
iter_nums = 1000

#初始化theta
theta = np.zeros((df.shape[1],1))

#标准化数据
df_1 = normalization(df)

X = np.array(df_1.iloc[:,:-1])
#show(X,settings={'block':True})

labels = np.array(df_1["param3"])
#show(df_1,settings={'block':True})

m = df.shape[0]

labels = np.expand_dims(labels,axis = 1)

X = np.concatenate((np.ones((m,1)),X),axis=1)

#画出迭代次数和loss曲线
#fig, ax = plt.subplots(figsize=(16, 9))
fig  = plt.figure(figsize = (16,9))

for alpha in alpha_candidate:
        final_theta,loss = grediant_decent(alpha,m,X,theta,labels)
        plt.plot(np.arange(iter_nums), loss, label = alpha)

plt.xlabel('epoches',fontsize =18)
plt.ylabel('cost_data',fontsize =18)

plt.legend(bbox_to_anchor=(1, 1), loc=2)
plt.title('cost_data vs learning rate',fontsize = 18)
plt.show()


