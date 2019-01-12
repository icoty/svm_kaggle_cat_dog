一、基于传统机器学习方法SVM对kaggle猫狗图片分类<br>
user:　　　　　yangyu <br>
repository：　https://github.com/icoty/svm_kaggle_cat_dog <br>
参考文献:　　　https://www.jianshu.com/p/e62834b99bad <br>
---
---
二、目录树 <br>
. <br>
|-- README.md <br>
|-- resizeData/  原始图片像素尺寸不一,调用接口格式化后存储至该目录 <br>
|-- train/       原始数据解压后存放位置, 下载链接：https://pan.baidu.com/s/13hw4LK8ihR6-6-8mpjLKDA 密码：dmp4  <br>
`-- train.py  <br>

三、运行结果展示  <br>
[yangyu@VM_0_3_centos svm]$ <br>
[yangyu@VM_0_3_centos svm]$ python train.py <br>
===============　[　　　　　　　　BEGIN　　　　　　　] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
                 2019-01-12 19:04:56<br>
                 2019-01-12 19:04:56<br>
trainData:[500],　cat:[245],　dog:[255],　testData:[50]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:04:56<br>
                 2019-01-12 19:04:56<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:04:56<br>
                 2019-01-12 19:04:56<br>
predict:[50], errorConuts:[25], accuracy:[50.00%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
=============== [        BEGIN       ] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
                 2019-01-12 19:04:56<br>
                 2019-01-12 19:04:57<br>
trainData:[1000],cat:[505],dog:[495],testData:[100]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:04:57<br>
                 2019-01-12 19:04:57<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:04:57<br>
                 2019-01-12 19:04:57<br>
predict:[100], errorConuts:[43], accuracy:[57.00%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
=============== [        BEGIN       ] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
                 2019-01-12 19:04:57<br>
                 2019-01-12 19:04:58<br>
trainData:[2000],cat:[1009],dog:[991],testData:[200]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:04:58<br>
                 2019-01-12 19:04:59<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:04:59<br>
                 2019-01-12 19:04:59<br>
predict:[200], errorConuts:[90], accuracy:[55.00%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
=============== [        BEGIN       ] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
                 2019-01-12 19:04:59<br>
                 2019-01-12 19:05:00<br>
trainData:[5000],cat:[2521],dog:[2479],testData:[500]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:05:00<br>
                 2019-01-12 19:05:02<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:05:02<br>
                 2019-01-12 19:05:02<br>
predict:[500], errorConuts:[244], accuracy:[51.20%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
=============== [        BEGIN       ] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
                 2019-01-12 19:05:02<br>
                 2019-01-12 19:05:08<br>
trainData:[15000],cat:[7516],dog:[7484],testData:[1500]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:05:08<br>
                 2019-01-12 19:05:12<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:05:12<br>
                 2019-01-12 19:05:12<br>
predict:[1500], errorConuts:[713], accuracy:[52.47%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
=============== [        BEGIN       ] ===============<br>
############### [Hog_SVM  pre process] ###############<br>
　　　　　　　　　　　　　　　　2019-01-12 19:05:12<br>
　　　　　　　　　　　　　　　　2019-01-12 19:05:21<br>
trainData:[25000],cat:[12500],dog:[12500],testData:[2500]<br>
############### [Hog_SVM  pre process] ###############<br>
############### [Hog_SVM       train ] ###############<br>
                 2019-01-12 19:05:21<br>
                 2019-01-12 19:05:27<br>
############### [Hog_SVM       train ] ###############<br>
############### [Hog_SVM     predict ] ###############<br>
                 2019-01-12 19:05:27<br>
                 2019-01-12 19:05:27<br>
predict:[2500], errorConuts:[1181], accuracy:[52.76%]<br>
############### [Hog_SVM     predict ] ###############<br>
---
---
[yangyu@VM_0_3_centos svm]$ ls <br>
