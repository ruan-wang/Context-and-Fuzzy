import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from Context1 import contex3
import random
from random import sample

#插值型隶属函数
def A1 (x):
    f=0
    if x>=830 and x<=1232:
        f1=0*x**2+0.00248756218905473*x-2.06467661691542
        f=f1
    if x>1232 and x<=1640:
        f2=0*x**2-0.00245098039215686*x+4.01960784313725
        f=f2
    return(f)
#A0
def A0 (x):
    f=0
    if x>=0 and x<=47:
        f1=0.0000192722787542399*x**2+0.0203707986432316*x+0
        f=f1
    if x>47 and x<=123:
        f2=0*x**2-0.0131578947368421*x+1.61842105263158
        f=f2
    return(f)

#G1
def G1 (x):
    f=0
    if x>=130 and x<=160:
        f1=0*x**2+0.0333333333333333*x-4.33333333333333
        f=f1
    if x>160 and x<=200:
        f2=0*x**2-0.0250000000000000*x+5.00000000000000
        f=f2
    return(f)

#G0
def G0 (x):
    f=0
    if x>=240 and x<=251:
        f1=0.00151515151515152*x**2-0.653030303030303*x+69.4545454545455
        f=f1
    if x>251 and x<=255:
        f2=0*x**2-0.250000000000000*x+63.7500000000000
        f=f2
    return(f)



#加载训练好的网络
#net = torch.load('network2\\net2.pth')
net = torch.load('network\\network H5\\net1.pth')










#填涂点数据化函数
def point(img):
    image=img
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#灰度
    _O, binaryimage = cv2.threshold(grayimage, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    gm=cv2.mean(grayimage)[0]
    area=len(binaryimage[binaryimage==255])
    #算出四个隶属值
    #A1
    a1=A1(area)
    #A0
    a0=A0(area)
    #G1
    g1=G1(gm)
    #G0
    g0=G0(gm)

    #隶属乘法器
    pai1=a1*g1
    pai2=a0*g0

    d=[]
    d.extend([pai1,pai2])
    return d



#网络输出
def OutPut(data):
    target = np.array(data)
    target = torch.from_numpy(target).float()

    target=(net(target))[0][0]
    return('%.20f' % target)


def Judge(img):
    #填涂选项数据化
        #读入图像
    image=img

        #四个填涂选项隶属数据化
    A=point(image[37:70,18:95])
    B=point(image[37:70,119:192])
    C=point(image[37:70,218:289])
    D=point(image[37:70,315:387])

        #保存四个选择的隶属数据
    dataA=[]
    dataA.append(A)
    dataB=[]
    dataB.append(B)
    dataC=[]
    dataC.append(C)
    dataD=[]
    dataD.append(D)


    #预测结果
    a=float(OutPut(dataA))
    b=float(OutPut(dataB))
    c=float(OutPut(dataC))
    d=float(OutPut(dataD))
    results=[]
    if a>=0.35:
        results.append('A')
    if a>=0.25 and a<0.35:
        k=contex3(image,1)
        if k==1:
            results.append('A')


    if b>=0.35:
        results.append('B')
    if b>=0.25 and b<0.35:
        k=contex3(image,2)
        if k==1:
            results.append('B')


    if c>=0.35:
        results.append('C')
    if c>=0.25 and c<0.35:
        k=contex3(image,3)
        if k==1:
            results.append('C')


    if d>=0.35:
        results.append('D')
    if d>=0.25 and d<0.35:
        k=contex3(image,4)
        if k==1:
            results.append('D')
    AA=sample(['A','B','C','D'], 1)
    if len(results)==0:
        results.append(AA[0])
    print(results)





##批量读入图像路径
Dir=[]
length=len(os.listdir("Filling-point-Set\Testset"))

#读入图片地址
for i in range(length):
    Dir.append("Filling-point-Set\Testset\{}.jpg".format(str(i+1))) 

#输出结果
for j in range(length):
    print("第",j+1,"张")
    image=cv2.imread(Dir[j])
    Judge(image)