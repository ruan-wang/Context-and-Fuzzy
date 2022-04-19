import cv2 as cv
import numpy as np


#欧式比较
def contrast(gm,area,aaa,bbb):
    k=0
    d=100

    if area>200 or gm<150:
        k=1
        d=0
    else:
        pdist1=((gm-aaa)**2)**0.5 
        pdist2=((area-bbb)**2)**0.5 
        d=(pdist1+pdist2)/2
        #精度控制
        #d=int("%.18f"%d)
        d=int(d)
        if d<=50:
            k=1
    return k




#上下文比较
def contex3(image,key):
    if key==1:
        a=18
        b=95
    if key==2:
        a=119
        b=192
    if key==3:
        a=218
        b=289
    if key==4:
        a=315
        b=387
    grayimage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)#灰度
    _O, binaryimage = cv.threshold(grayimage, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)

    #选项特征提取
    gm=cv.mean(grayimage[37:70,a:b])[0]#裁剪坐标为[y0:y1, x0:x1]
    binaryimage111=binaryimage[37:70,a:b]
    area=len(binaryimage111[binaryimage111==255])


    #所有选项特征
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
    if key==1:
        A=[gm2,gm3,gm4]#灰度越小越真
        A1=sorted(A)
        A2=[A.index(A1[0]),A.index(A1[1]),A.index(A1[2])]#升序排列的A的下标
        B=[area2,area3,area4]#面积越大越真。做一个copy数组A1,用来降序并在原数组中找到对应的index,A2保存index
        B1=sorted(B,reverse = True)
        B2=[B.index(B1[0]),B.index(B1[1]),B.index(B1[2])]#降序排列的B的下标
        #print(gm,area,A,A2,B,B2)
        aa=A[A2[0]]
        bb=B[B2[0]]
        k=contrast(gm,area,aa,bb)
    if key==2:
        A=[gm1,gm3,gm4]#灰度越小越真
        A1=sorted(A)
        A2=[A.index(A1[0]),A.index(A1[1]),A.index(A1[2])]#升序排列的A的下标
        B=[area1,area3,area4]#面积越大越真。做一个copy数组A1,用来降序并在原数组中找到对应的index,A2保存index
        B1=sorted(B,reverse = True)
        B2=[B.index(B1[0]),B.index(B1[1]),B.index(B1[2])]#降序排列的B的下标
        #print(gm,area,A,A2,B,B2)
        aa=A[A2[0]]
        bb=B[B2[0]]
        k=contrast(gm,area,aa,bb)
    if key==3:
        A=[gm1,gm2,gm4]#灰度越小越真
        A1=sorted(A)
        A2=[A.index(A1[0]),A.index(A1[1]),A.index(A1[2])]#升序排列的A的下标
        B=[area1,area2,area4]#面积越大越真。做一个copy数组A1,用来降序并在原数组中找到对应的index,A2保存index
        B1=sorted(B,reverse = True)
        B2=[B.index(B1[0]),B.index(B1[1]),B.index(B1[2])]#降序排列的B的下标
        #print(gm,area,A,A2,B,B2)
        aa=A[A2[0]]
        bb=B[B2[0]]
        k=contrast(gm,area,aa,bb)
    if key==4:
        A=[gm1,gm2,gm3]#灰度越小越真
        A1=sorted(A)
        A2=[A.index(A1[0]),A.index(A1[1]),A.index(A1[2])]#升序排列的A的下标
        B=[area1,area2,area3]#面积越大越真。做一个copy数组A1,用来降序并在原数组中找到对应的index,A2保存index
        B1=sorted(B,reverse = True)
        B2=[B.index(B1[0]),B.index(B1[1]),B.index(B1[2])]#降序排列的B的下标
        #print(gm,area,A,A2,B,B2)
        aa=A[A2[0]]
        bb=B[B2[0]]
        k=contrast(gm,area,aa,bb)
    return k











