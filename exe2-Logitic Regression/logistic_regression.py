import os
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from pandasgui import show
from sklearn.metrics import classification_report#这个包是评价报告
import  scipy.optimize as opt

def get_x(df):#读取特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return np.array(data.iloc[:, :-1]) 


def get_y(df):#读取标签
    return np.array(df.iloc[:, -1])#df的最后一列


def normolization(df):#特征缩放
        return df.apply(lambda x: (x-x.mean())/x.std())

def sigmoid(x):
        return 1/(1+np.exp(-x))

#代价计算函数，这里ndarray和matrix需要注意一下区别，如果是matrix需要使用np.multiply()
def compute_cost(theta,x,y):
        return np.mean(-y*np.log(sigmoid(x.dot(theta)))-(1 - y)*np.log(1-sigmoid(x.dot(theta))))

#梯度计算函数
def gradient(theta,x,y):
#        theta = np.reshape(theta,(x.shape[1],1))
        return (1/len(x))*x.T.dot(sigmoid(x.dot(theta))-y)

#预测函数
def predict_res(theta,x):
        pre = sigmoid(x.dot(theta))
        return (pre>=0.5).astype(int)



#读取数据
root_dir = '/home/leedo/ML_Andrew_Ng/exe/exe2'
df = pd.read_csv(os.path.join(root_dir,'ex2data1.txt'),names = ['exam1','exam2','result'])
#df.head()
x_1= get_x(df)
y_1 = get_y(df)

#初始化参数
theta = np.zeros(3)
#利用scipy中的minimize或者fmin_tnc来计算代价和梯度参数，这里需要注意参数的传递，即代价计算函数和梯度计算函数中参数的顺序需要和minimize中一致
#res = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient, args=(x, y))
res = opt.minimize(fun=compute_cost, x0=theta, args=(x_1, y_1), method='Newton-CG', jac=gradient)
theta_last = res.x

#print(compute_cost(theta_last,x_1,y_1))

#print(res)


#绘制结果
fig = plt.figure(figsize = (16,9))
sns.scatterplot(x='exam1', y='exam2', hue='result', data=df)

x = np.arange(130,step = 0.1)
y = -theta_last[0]/theta_last[2]-theta_last[1]/theta_last[2]*x
plt.plot(x, y, 'grey')
#plt.xlim(0, 130)
#plt.ylim(0, 130)
plt.title('Boundary')
plt.show()


                                                                ########################################################################
                                                                #########################                                                  ###########################
                                                                #########################           正则化逻辑回归        ###########################
                                                                #########################                                                  ###########################
                                                                ########################################################################
                                                                

#可视化数据
df_2 = pd.read_csv(os.path.join(root_dir,'ex2data2.txt'),names = ['Test1','Test2','Accepted']) 

positive = df_2[df_2['Accepted'].isin([1])]
negative = df_2[df_2['Accepted'].isin([0])]

fig, ax = plt.subplots(figsize=(12,8))
ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
ax.legend()
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')
plt.show()

##正则化代价计算，不惩罚theta0
def compute_cost_nor(theta,x,y,lamb):
        return np.mean(-y*np.log(sigmoid(x.dot(theta)))-(1 - y)*np.log(1-sigmoid(x.dot(theta))))+(lamb/(2*len(x)))*np.sum(pow(theta[1:],2))

#由于数据没有线性的决策边界，因此使用多项式
def mapFeature(x,y,power):
        data = {"F{}{}".format(i-j,j):pow(x,i-j)*pow(y,j) for i in range(power+1) for j in range(i+1)}
        return data

#正则化梯度计算
def gradient_nor(theta,x,y,lamb):
        grad = np.zeros(x.shape[1])
        grad_tmp = (1/len(x))*x.T.dot(sigmoid(x.dot(theta))-y)
        grad[0]  = grad_tmp[0]  
        grad[1:] =  grad_tmp[1:]+(lamb/len(x))*theta[1:]
        return  grad

x_2 = df_2['Test1']
y_2 = df_2['Test2']

power = 6
df_3 = pd.DataFrame(mapFeature(x_2,y_2,power))

#df_3.insert(0,'Accepted',df_2['Accepted'])
show(df_3,settings={'block':True})
x_nor = np.array(df_3)
y_nor = get_y(df_2)

##初始参数
theta_nor = np.zeros(df_3.shape[1])
##代价系数
lamb = 1

#参数拟合
res_nor = opt.minimize(fun=compute_cost_nor, x0=theta_nor, args=(x_nor, y_nor,lamb), method='Newton-CG', jac=gradient_nor)
theta_res = res_nor.x
print(res_nor)

#进行预测准确率分析
y_pred = predict_res(theta_res,x_nor)
print(classification_report(y_nor, y_pred))
