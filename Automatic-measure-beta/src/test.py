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
data1=pd.read_csv('C:/Wail/automatic-measure-of-imitation/Automatic-measure-beta/Data/vid-gen2/wave1.txt',header=None, sep=r"\s+",skiprows=3)
data2=pd.read_csv('C:/Wail/automatic-measure-of-imitation/Automatic-measure-beta/Data/vid-gen1/wave1.txt',header=None, sep=r"\s+",skiprows=3)




#droping useless columns
data1.drop(data1.columns[[0,1,2,4,5,6]], axis=1, inplace=True)
data2.drop(data2.columns[[0,1,2,4,5,6]], axis=1, inplace=True)


#naming the columns
data1.columns=column
data2.columns=column



data10=data1[(data1.frame>=max(min(data1.frame),min(data2.frame))) & (data1.frame<=min(max(data1.frame),max(data2.frame)))]
data20=data2[(data2.frame>=max(min(data1.frame),min(data2.frame))) & (data2.frame<=min(max(data1.frame),max(data2.frame)))]


#merging the two dataframes
data=data1.append(data2,ignore_index=True)
data=data.drop(['frame'],axis=1)

data0=data10.append(data20,ignore_index=True)
datapca0=data0.drop(['frame'],axis=1)



# scaling zero mean and unit variance or maybe add normalization
# in numpy 
hoghof_scaled = preprocessing.scale(datapca)
hoghof_scaled0 = preprocessing.scale(datapca0)


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
    
    print(ncomponent)    
    
    if ncp:
        ncomponent=ncp
        
        
    
    return X[:,0:ncomponent],ncomponent
    

hoghofPCA,ncomponent=applypca(hoghof_scaled)
hoghofPCA0,ncomponent0=applypca(hoghof_scaled0)

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
kmean=k_means_predict(data,5)
kmean0=k_means_predict(hoghofPCA0,256)

  
kmean.get_params()  
  
  
##############################generate training test cross validation data for one SVM ###########################################
  
  
 #list of non duplicated frame number 
 def histograms(data,K):
     i=0
     gbp=data
     predclust=kmean.predict(preprocessing.scale((data.drop(['frame'],axis=1))))
     gbp['pred']=predclust
     gbp=gbp.groupby('frame')
     train=np.zeros((len(gbp),K))
     for name,group in gbp:
         train[i][group.pred.values]=1
         #print(train[i])
         i+=1
         
     return train
         
 #104 is ncomponent and 256 is ncluster K
 h2=histograms(data2,5)
 h1=histograms(data1,5)     
     
    
    
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

dec=svm1.predict(h1)
SAB(h1,h2,4,4)

def recurrence_matrix(h1,h2,threshold,i,j):
    
    return np.where(threshold-SAB(h1,h2,i,j) > 0 ,1, 0)

h1perc=float(h1.shape[0])/float((h1.shape[0]+h2.shape[0]))
h2perc=1-h1perc
    
    
onesvm = svm.OneClassSVM( kernel=my_kernel, nu=0.00300)
onesvm.fit(h2)
np.mean(onesvm.predict(h2)==1)


from sklearn.cross_validation import KFold
it=np.linspace(0.001, 1, 1000)

it[999]

def Cross_validation(h):
    n_folds=4
    nu=np.linspace(0.001, 1, 1000)
    results=[]
    for d in nu:
        onesvm = svm.OneClassSVM( kernel=my_kernel, nu=d)
        hypothesisresults=[]
        for train, test in KFold(16, n_folds):
            onesvm.fit(h[train]) # fit
            hypothesisresults.append(np.mean(onesvm.predict(h[test])==1))
            
        results.append(np.mean(hypothesisresults))
        #print(results)
        
        
    return nu[np.argmax(results)]
    
 len(h2)   

Cross_validation(h2)


dec=(onesvm.decision_function(h2)-onesvm.intercept_)*h1perc
dec2=(onesvm.decision_function(h1)-onesvm.intercept_)*h1perc
print(sum(dec))
print(sum(dec2))


onesvm = svm.OneClassSVM( kernel=my_kernel, nu=0.1)
onesvm.fit(h2)
dec=(onesvm.decision_function(h2)-onesvm.intercept_)*h2perc
dec2=(onesvm.decision_function(h1)-onesvm.intercept_)*h2perc
print(sum(dec))
print(sum(dec2))



def cv_optimize(clf, parameters, X, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        
    gs.fit(X)
    print "BEST", gs.best_params_, gs.best_score_, gs.grid_scores_
    best = gs.best_estimator_
    return best





onesvm=svm.OneClassSVM( kernel='precomputed')
nu_param=np.linspace(0.001, 1, 200)
yh2=np.ones(h2.shape[0])
yh1=np.ones(h1.shape[0])

parameters=dict(nu=nu_param)
gs = GridSearchCV(onesvm, param_grid=parameters, n_jobs=1, cv=10,scoring='f1') 
precomputedK=my_kernel(h2,h2)
gs.fit(precomputedK,yh2)
prr=gs.best_params_   
    

    


   

     
 
 
 
 

   
    

    
    
    
    
    
    
    


  
  
  
  

  
  
  

  
  



    







