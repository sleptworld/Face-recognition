from PIL import Image
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import logging

class test:
    def __init__(self,path,trainNum,size):
        self.path = path
        self.files = os.listdir(self.path)
        self.filenum = len(self.files)
        self.trainNum = trainNum
        self.d_train = None
        self.engineface = None
        try:
            assert isinstance(size,tuple)
            self.imgSize = size
        except AssertionError:
            print("请输入正确类型")

    def __getData(self,files,end,begin=0):
        data = []
        for i in range(begin,end):
            file = files[i]
            img = Image.open(self.path+file).resize(size=(self.imgSize[1],self.imgSize[0]))
            temp = np.array(img).flatten()
            data.append(temp)
        return np.array(data)

    def getTrainData(self):
        return self.__getData(self.files,self.trainNum)
    def getTestData(self):
        return self.__getData(self.files,self.filenum,self.trainNum)

    def getMain(self,X,num):
        X = scale(X)
        pca = PCA()
        pca.fit(X)
        sum = 0
        c = 0
        for i in pca.explained_variance_ratio_:
            if sum >= num:
                print("降至：{}维".format(c+1))
                break
            else:
                sum += i
                c += 1
    def meanFace(self,X):
        temp = np.mean(X,axis=0)
        p = Image.fromarray(np.reshape(temp,self.imgSize))
        p.show()

    def down(self,num,X):
        X = scale(X)
        pca = PCA(num)
        pca.fit(X)
        self.d_train = pca.transform(X)
        print(pca.components_[0].shape)
        return self.engineface