import cv2
import numpy as np
import os.path
import pandas as pd

def image_colorfulness(image):
    # 将图片划分为R，G，B三色通道（这里得到的R，G，B喂向量而非标量）
    (B,G,R) = cv2.split(image.astype("float"))

    # rg = R - G
    rg = np.absolute(R - G)

    # yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # 计算rg和yb的平均值和标准差
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # 计算rgyb的标准差和平均值
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # 返回颜色丰富度C（范围 0-100）
    return stdRoot + (0.3 * meanRoot)


def contrast(img0):
    # 将图片转化为灰度图片
    img1 = cv2.cvtColor(img0,cv2.COLOR_BGR2GRAY)
    m,n = img1.shape
    # 图片矩阵向外扩散一个像素方便计算,除以1.0的目的是转化为float类型，便于后续计算
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 + (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)
    
    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4)
    return cg


def HSVcompute(imageHSV):
    hsv = cv2.cvtColor(imageHSV ,cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)
    # 取色度非0的值
    h = H.ravel()[np.flatnonzero(H)]
    # 计算平均色度
    average_h = sum(h)/len(h)
    # print("色度为："+str(average_h))
    # 取亮度非0的值
    v = V.ravel()[np.flatnonzero(V)]
    # 计算平均亮度
    average_v = sum(v)/len(v)
    # print("饱和度为："+str(average_v))
    # 取饱和度非0的值
    s = S.ravel()[np.flatnonzero(S)]
    # 计算平均饱和度
    average_s = sum(s)/len(s)
    # print("亮度为："+str(average_s))
    return average_h, average_s, average_v