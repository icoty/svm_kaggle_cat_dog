# !/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import time
import cv2 as cv
import numpy as np
from glob import glob
from os.path import dirname, join, basename

def printCurTime():
    print('\t\t',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    
"""
基于传统机器学习方法SVM对图片分类
user:        yangyu
repository： https://github.com/icoty/svm_kaggle_cat_dog
数据集：     下载链接：https://pan.baidu.com/s/13hw4LK8ihR6-6-8mpjLKDA 密码：dmp4
"""
# 基于传统机器学习方法SVM对图片分类
class Hog_SVM(object):
    def __init__(self, trainDataPath = './train', resizePath = './resizeData', hoRatio = 0.10):
        self.train_path = trainDataPath
        self.resize_path = resizePath
        self.hoRatio    = hoRatio           # 测试集比率
        self.bin_n      = 16*16             # Number of bins
        self.trainData  = []
        self.trainLabel = []
        self.testData   = []   
        self.testLabel  = []
        
    def resize(self):
        print("############### [Hog_SVM resize image] ###############")
        printCurTime()
        
        num = 0
        for fn in glob(join(self.train_path, '*.jpg')):
            img = cv.imread(fn)
            res = cv.resize(img,(64,128),interpolation=cv.INTER_AREA)
            if "cat" in fn:
                cv.imwrite(self.resize_path +'/cat_' + str(num)+'.jpg',res)
            else:
                cv.imwrite(self.resize_path +'/dog_' + str(num)+'.jpg',res)
            num = num+1
        cv.waitKey(0)
        cv.destroyAllWindows()   
        
        printCurTime()   
        print("############### [Hog_SVM resize image] ###############\n")

    def read_samples(self, limit = 25000): # 默认读取25000的数据
        print("=============== [        BEGIN       ] ===============")
        print("############### [Hog_SVM  pre process] ###############")
        printCurTime()
        
        img = {}
        cat = 0
        total = 0
        for fn in glob(join(self.resize_path, '*.jpg')):
            if total < limit:
                #print(fn)
                img[total] = cv.imread(fn,0)  #参数加0，只读取黑白数据，去掉0，就是彩色读取。
                total = total+1
                if "cat" in fn:
                    cat = cat + 1
                    self.trainLabel.append(1)
                else:
                    self.trainLabel.append(-1)
                
        trainpic =[]
        for i in img:
            trainpic.append(img[i])
        
        hogdata = list(map(self.Hog,trainpic))
        self.trainData = np.float32(hogdata).reshape(-1,self.bin_n*4)
        
        numTest = int(total*self.hoRatio)
        self.testData = self.trainData[0:numTest,:]
        self.testLabel = self.trainLabel[0:numTest]
        self.trainData = self.trainData[numTest:,:]
        self.trainLabel = self.trainLabel[numTest:] 
        #print('\t\t\t trainData:\t',np.array(self.trainData).shape)
        #print('\t\t\t trainLabel:\t',np.array(self.trainLabel).shape)
        #print('\t\t\t testData:\t',np.array(self.testData).shape)
        #print('\t\t\t testLabel:\t',np.array(self.testLabel).shape)
        printCurTime()    
        print("trainData:[%d],cat:[%d],dog:[%d],testData:[%d]" % (total, cat, total - cat, total*self.hoRatio))
        print("############### [Hog_SVM  pre process] ###############")
        
    def Hog(self,img):
        x_pixel,y_pixel=194,259
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bins = np.int32(self.bin_n*ang/(2*np.pi))    # quantizing binvalues in (0...16)
        bin_cells = bins[:x_pixel//2,:y_pixel//2], bins[x_pixel//2:,:y_pixel//2], bins[:x_pixel//2,y_pixel//2:], bins[x_pixel//2:,y_pixel//2:]
        mag_cells = mag[:x_pixel//2,:y_pixel//2], mag[x_pixel//2:,:y_pixel//2], mag[:x_pixel//2,y_pixel//2:],mag[x_pixel//2:,y_pixel//2:]
        hists = [np.bincount(b.ravel(), m.ravel(), self.bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)     # hist is a 64 bit vector
        return hist  
    
        #svm参数配置
    def svm_config(self, name='./svm_cat_dog.model'):
        print("############### [Hog_SVM       train ] ###############")
        printCurTime()
        
        svm = cv.ml.SVM_create()
        criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 1000, 1e-3)
        svm.setTermCriteria(criteria)
        svm.setGamma(5.383)
        svm.setKernel(cv.ml.SVM_LINEAR)
        svm.setNu(0.5)
        svm.setP(0.1)
        svm.setC(0.01)
        svm.setType(cv.ml.SVM_C_SVC)
        self.svm_train(svm)
        self.svm_save(svm, name)
        
        printCurTime()    
        print("############### [Hog_SVM       train ] ###############")
        return svm

    #svm训练
    def svm_train(self,svm):
        svm.train(np.array(self.trainData),cv.ml.ROW_SAMPLE,np.array(self.trainLabel, dtype = np.int32))
        
    #svm参数保存
    def svm_save(self, svm, name='svm_cat_dog.model'):
        svm.save('svm_cat_dog.model')

    #svm加载参数 
    def svm_load(svm, name = './svm_cat_dog.model'):
        svm = cv.ml.SVM_load(name)
        return svm

    def svm_predict(self, svm, test_data = [], test_label = []):
        print("############### [Hog_SVM     predict ] ###############")
        printCurTime()
        
        if 0 != len(test_data) and 0 != len(test_label):
            self.testLabel = test_label
            self.testData = test_data
            
        result = svm.predict(self.testData)
        m = np.shape(self.testLabel)[0]
        errorCount = 0.0
        for i in range(m):
            if int(self.testLabel[i//1]) != int(result[1][int(i)][0]) :
                errorCount = errorCount + 1
        
        printCurTime()   
        print("predict:[%d], errorConuts:[%d], accuracy:[%.2f%%]" % (m,errorCount,(float(m-errorCount)/m)*100))
        print("############### [Hog_SVM     predict ] ###############\n")
        
if __name__ == '__main__':
    # 数据量分别取[500, 1000, 2000, 5000, 15000, 25000]
    for i in [500, 1000, 2000, 5000, 15000, 25000]:
        hog_svm = Hog_SVM('./train', './resizeData', 0.10)
        #hog_svm.resize()  #机器上首次执行时开启, 格式化图片仅需一次
        hog_svm.read_samples(i)
        svm = hog_svm.svm_config()
        hog_svm.svm_predict(svm)