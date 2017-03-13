# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:04:13 2017

@author: isir
"""

class Imitation:
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
    import sys, os
    
    #Class variables 
     #STIP: Space-Time Interest Points detection
    pathSTIP="\\Automatic-measure-beta\\util\\cvml10-actions\\cvml10-actions\\stip\\bin\\stipdet.exe "
     # The folders where the STIP are stored for each video  
    pathout1="\\Automatic-measure-beta\\Data\\vid-gen1\\"
    pathout2="\\Automatic-measure-beta\\Data\\vid-gen2\\"


    
#    im=Imitation("walk-simple","walk-complex",256,'C:\\Wail')  im.load_process_data()
    
    
    def __init__(self,vid1Name,vid2Name,K,projectPath,windowsize=1,skip=1,scale=1,pca=1):
        """
        -init:
            vid1Name: the name of the video located in the Data folder
            K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
            projectPath: Where the project is located in your computer example "C:\\Wail"
            windowsize: number of frames for each training vector  example of 2 frame : vect[1,2]--vect[3,4]--vect[5,6]
            skip: skip generating STIP points if they are allready generated
            pca: apply PCA or not
        """
        
        self.vid1Name=vid1Name
        self.vid2Name=vid2Name
        self.K=K
        self.windowsize=windowsize
        self.projectPath=projectPath
        self.skip=skip
        self.scale=scale
        self.pca=pca
        
        
    def compute(self):
        data1,data2,hoghof=self.load_process_data()
        
      
        hoghof,ncomponent=self.applypca(hoghof)
            
        kmean=self.k_means_predict(hoghof,self.K)
        h2=self.histograms(data2,ncomponent,self.K)
        h1=self.histograms(data1,ncomponent,self.K) 
        print(h1)
        print(h2)
        

            
        
            
        
        
        
        
        
    def applypca(data,ncp=None):
        """
        input:
            -data: hoghof descriptors
        output:
            - principal component with variance 99%
        description: Dimension reduction with 99% of variance  
        """
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
            
        print("######## Done applying PCA###############")
        
    
        return X[:,0:ncomponent],ncomponent 
    
    
    def histograms(data,ncomponent,K):
        """
        -input:
            -data: hoghof descriptor
            -number of principal component (comes from applying pca) 
            -K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
        """
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
           
        print("######### Generating histograms ###############")
         
        return train
    
    
    def k_means_predict(data,K,ini='k-means++'):
        """
        input:
            -data: hoghf descriptor
        output:
            -Kmean classifier
        descrition: apply Kmeans
        """
        t0 = time()
        kmean= KMeans(init=ini, n_clusters=K, n_init=10) 
        kmean.fit(data)
        t=time()
        print("training  time for kmeans------:"+str(t-t0))
        print("silhouette ------:"+str(metrics.silhouette_score(data, kmean.labels_,metric='euclidean')))
        return kmean
    
    
    def my_kernel(X,Y):
    
        return X.dot(Y.T)
    
    
    def OneSVM_predict(h,my_kernel):
        t0 = time()
        onesvm = svm.OneClassSVM( kernel=my_kernel)
        onesvm.fit(h1)
        t=time()
        print("training time for svm----------: "+str(t-t0))
        return onesvm
    
    
    
    def SAB(h1,h2,i,j):
        svm1=OneSVM_predict(h1,my_kernel)
        svm2=OneSVM_predict(h2,my_kernel)
        Sab1=svm1.decision_function(h2)
        Sab2=svm2.decision_function(h1)
        return Sab1[i]+Sab2[j]
    
    
    
    def recurrence_matrix(h1,h2,threshold,i,j):
    
        return np.where(threshold-SAB(h1,h2,i,j) > 0 ,1, 0)




    def load_process_data(self):
        """
        output: 
            -data1: dataframe for the first videoc 'STIP' points
            -data2: dataframe for the second video 'STIP' points
            -hoghof: descriptor of both videos. We used a combination of  histogram of gradients and histogram of flow 
        """
        
        print("################ loading data ######################")
        
        pathVid1="\\Automatic-measure-beta\\Data\\"+self.vid1Name+".avi"
        pathVid2="\\Automatic-measure-beta\\Data\\"+self.vid2Name+".avi"


        # Generate a file .txt that contains STIP points  
        if self.skip!=1:
            os.system(self.projectPath+Imitation.pathSTIP+" -f "+self.projectPath+pathVid1+" -o "+self.projectPath+Imitation.pathout1+self.vid1Name+".txt -vis no")
            os.system(self.projectPath+Imitation.pathSTIP+" -f "+self.projectPath+pathVid2+" -o "+self.projectPath+Imitation.pathout2+ self.vid2Name+".txt -vis no")
            
        
        #loading the STIP into a panda dataframe
        data1=pd.read_csv('../Data/vid-gen1/'+self.vid1Name+'.txt',header=None, sep=r"\s+",skiprows=3)
        data2=pd.read_csv('../Data/vid-gen2/'+self.vid1Name+'.txt',header=None, sep=r"\s+",skiprows=3)
        
        #drop useless columns
        data1.drop(data1.columns[[0,1,2,3,5,6]], axis=1, inplace=True)
        data2.drop(data2.columns[[0,1,2,3,5,6]], axis=1, inplace=True)
        
        #columns names
        column=["hgf"+str(i) for i in range(162)]
        column.insert(0,'frame')
        #naming the columns
        data1.columns=column
        data2.columns=column
        
        #merging the two dataframes
        data=data1.append(data2,ignore_index=True)
        datapca=data.drop(['frame'],axis=1)
        
        if self.scale==1:
            hoghof = preprocessing.scale(datapca)
            
        else:
            #dataframe to numpy without scaling to be usable in sklearn
            hoghof=datapca.as_matrix(columns=None)
            
        print("############ loading data completed ###########################")
        print(hoghof)
            
        return data1,data2,hoghof
    
    
 
    
    
    

            
        
        
        
        
        
        
        
        
        
        
        
        
    