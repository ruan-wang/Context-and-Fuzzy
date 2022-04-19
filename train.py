import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
#训练集

with open("Label\\datalist.data","rb") as filehandle:
    datalist=pickle.load(filehandle)
data = np.array(datalist)


corpus = pd.read_csv('Label\\all_y_trues.csv',header=0, names=["y_trues"]) # 得到 DataFrame
corpus = np.array(corpus)  # 转换为 ndarray [[1], [2], [3]]
corpus = corpus.reshape(1, len(corpus)).tolist()  # 转换成 List [[1, 2, 3]]
corpus = corpus[0]  # 取第一个元素得到最终结果 [1, 2, 3]
corpus = np.array(corpus)
all_y_trues = np.array(corpus)

print(data)
print(all_y_trues)
data = torch.from_numpy(data).float()
all_y_trues = torch.from_numpy(all_y_trues).float()




#定义网络
net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 2),#输入层，两个神经元，隐含层也是两个
    #torch.nn.Sigmoid(),    #激活函数
    torch.nn.ReLU(),
    torch.nn.Linear(2, 1),#输出层两个神经元
    torch.nn.Softplus()
    )


# 训练部分
optimizer = torch.optim.SGD(net2.parameters(), lr=0.1)    #梯度下降方法
loss_function = torch.nn.MSELoss()   #计算误差方法

epoch_list=[]
loss_list=[] 
for i in range(1000):
    prediction = net2(data)
    prediction = prediction.squeeze(-1)
    loss = loss_function(prediction, all_y_trues)
    optimizer.zero_grad()    #消除梯度
    loss.backward()    #反向传播
    optimizer.step()     #执行
    if i%10==0:
        print("epoch=",i,"loss=",loss.data)
        epoch_list.append(i) # 插入迭代次数
        loss_list.append(loss.data) # 插入损失值

#画图
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(epoch_list,loss_list)
plt.show()

#保存模型
torch.save(net2, 'network\\net1.pth')