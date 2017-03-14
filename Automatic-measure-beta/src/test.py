# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 15:39:59 2017

@author: isir
"""
from sklearn.decomposition import PCA
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from multiprocessing import Pool
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from time import time



import numpy as np 
import pandas as pd


 ###################################Data preprocessing#####################################        
# colums name frame and hoghof descriptor
column=["hgf"+str(i) for i in range(162)]
column.insert(0,'frame')

#reading txt fi=pd.read_csv('../Data/vid-gen1/walk-complex.txt',header=None, sep=r"\s+")
data1=pd.read_csv('../Data/vid-gen1/walk-complex.txt',header=None, sep=r"\s+")
data2=pd.read_csv('../Data/vid-gen2/walk-simple.txt',header=None, sep=r"\s+")




#droping useless columns
data1.drop(data1.columns[[0,1,2,3,5,6]], axis=1, inplace=True)
data2.drop(data2.columns[[0,1,2,3,5,6]], axis=1, inplace=True)


#naming the columns
data1.columns=column
data2.columns=column

#merging the two dataframes
data=data1.append(data2,ignore_index=True)
datapca=data.drop(['frame'],axis=1)


# scaling zero mean and unit variance or maybe add normalization
# in numpy 
hoghof_scaled = preprocessing.scale(datapca)

#dataframe to numpy without scaling to be usable in sklearn
hoghof=datapca.as_matrix(columns=None)


################################PCA dimension reduction#############################################
#applying pca to reduce dimension
def applypca(data,ncp=None):
    pca = PCA(n_components=162)
    X = pca.fit_transform(data)

    expvar=pca.explained_variance_ratio_*100


    for i in range(len(expvar)+1):
        if sum(expvar[:i]) > 99:
            print i
            var= sum(expvar[:i])
            ncomponent=i
            break
        
    if ncp:
        ncomponent=ncp
        
    
    return X[:,0:ncomponent],ncomponent
    

    hoghofPCA,ncomponent=applypca(hoghof_scaled)
###############################Kmeans & code#######################################################################

#given by user vocabulary size and also class var
K=256

# evaluate clusters for different initialization
def k_means_predict(data,K,ini='k-means++'):
    t0 = time()
    kmean= KMeans(init=ini, n_clusters=K, n_init=10) 
    kmean.fit(data)
    t=time()
    print("training  time for kmeans------:"+str(t-t0))
    print("silhouette ------:"+str(metrics.silhouette_score(data, kmean.labels_,metric='euclidean')))
    return kmean

    
# class var 
kmean=k_means_predict(hoghofPCA,256)
  
kmean.get_params()  
  
  
##############################generate training test cross validation data for one SVM ###########################################
  
  
 #list of non duplicated frame number 
 def histograms(data,ncomponent,K):
     i=0
     gbp=data
     predclust=kmean.predict(applypca(preprocessing.scale((data.drop(['frame'],axis=1))),ncomponent)[0])
     gbp['pred']=predclust
     gbp=gbp.groupby('frame')
     train=np.zeros((len(gbp),K))
     for name,group in gbp:
         train[i][group.pred.values]=1
         print(train[i])
         i+=1
         
     return train
         
 #104 is ncomponent and 256 is ncluster K
 h2=histograms(data2,104,256)
 h1=histograms(data1,104,256)     
     
    
    
#######################################Training one class SVM SVM ##########################################################################

def my_kernel(X,Y):
    
    return X.dot(Y.T)


def OneSVM_predict(h,my_kernel):
    t0 = time()
    onesvm = svm.OneClassSVM( kernel=my_kernel)
    onesvm.fit(h1)
    t=time()
    print("training time for svm----------: "+str(t-t0))
    return onesvm

svm1=OneSVM_predict(h1,my_kernel)
#svm2=OneSVM_predict(h2,my_kernel)
#svm2.predict(h1)

#h1 and h2

def SAB(h1,h2,i,j):
    svm1=OneSVM_predict(h1,my_kernel)
    svm2=OneSVM_predict(h2,my_kernel)
    Sab1=svm1.decision_function(h2)
    Sab2=svm2.decision_function(h1)
    return Sab1[i]+Sab2[j]


SAB(h1,h2,4,4)

def recurrence_matrix(h1,h2,threshold,i,j):
    
    return np.where(threshold-SAB(h1,h2,i,j) > 0 ,1, 0)
    
    
onesvm = svm.OneClassSVM( kernel=my_kernel)
onesvm.fit(data1)
onesvm.intercept_
onesvm.



    


    
    

    


   

     
 
 
 
 

   
    

    
    
    
    
    
    
    


  
  
  
  

  
  
  

  
  



    







