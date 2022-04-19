import cv2 as cv
import imutils
import numpy as np
'''#四个参数欧氏距离
def pdist(x1,x2,y1,y2):
    #d=(((x1-x2)/255)**2+(y1-y2)**2)**0.5
    d=((x1-x2)**2+(y1-y2)**2)**0.5
    return d'''
#两个参数的欧氏距离
def pdist(x1,x2):
    #d=(((x1-x2)/255)**2+(y1-y2)**2)**0.5
    d=((x1-x2)**2)**0.5
    return d


image=cv.imread("datasetmaker\\image\\model2.jpg")
grayimage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#灰度
_O, binaryimage = cv.threshold(grayimage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

#特征提取
gm1=cv.mean(grayimage[37:70,18:95])[0]#裁剪坐标为[y0:y1, x0:x1]
gm2=cv.mean(grayimage[37:70,119:192])[0]
gm3=cv.mean(grayimage[37:70,218:289])[0]
gm4=cv.mean(grayimage[37:70,315:387])[0]

binaryimage1=binaryimage[37:70,18:95]
binaryimage2=binaryimage[37:70,119:192]
binaryimage3=binaryimage[37:70,218:289]
binaryimage4=binaryimage[37:70,315:387]
area1=len(binaryimage1[binaryimage1==255])
area2=len(binaryimage2[binaryimage2==255])
area3=len(binaryimage3[binaryimage3==255])
area4=len(binaryimage4[binaryimage4==255])



#上下文对比算法
A=[gm1,gm2,gm3,gm4]#灰度越小越真
A1=sorted(A)
A2=[A.index(A1[0]),A.index(A1[1]),A.index(A1[2]),A.index(A1[3])]
B=[area1,area2,area3,area4]#面积越大越真。做一个copy数组A1,用来降序并在原数组中找到对应的index,A2保存index
B1=sorted(B,reverse = True)
B2=[B.index(B1[0]),B.index(B1[1]),B.index(B1[2]),B.index(B1[3])]#降序排列的B的下标



X=[]
X.append(B2[0])
for i in range(0,3):
    if B[B2[i+1]]>400:
        X.append(B2[i+1])
        #f分段阈值加相似度比较
    if B[B2[i+1]]<=400:
        if i==0:
            break
        if i<=1:
            d=pdist(A[B2[i]],A[B2[i+1]])
            if d<15:
                X.append(B2[i+1])
        if i==2:
            d=pdist(A[B2[i]],A[B2[i+1]])
            if d<8:
                X.append(B2[i+1])
        





#将数字转成选项
Y=[]
if 0 in X:
    Y.append('A')
if 1 in X:
    Y.append('B')
if 2 in X:
    Y.append('C')
if 3 in X:
    Y.append('D')

print("推测填涂选项为：",Y)

cv.imshow("src",image)
cv.waitKey(0)



