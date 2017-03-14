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


#    
#    im=Imitation("walk-complex","walk-simple",256,'C:\\Wail\\automatic-measure-of-imitation',skip=1)
#    im.compute()
#    
    
    def __init__(self,vid1Name,vid2Name,K,projectPath,windowsize=1,skip=1,scale=1,pca=1,threshold=0.1):
        """
        -init:
            vid1Name: the name of the video located in the Data folder
            K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
            projectPath: Where the project is located in your computer example "C:\\Wail"
            windowsize: number of frames for each training vector  example of 2 frame : vect[1,2]--vect[3,4]--vect[5,6]
            skip: skip generating STIP points if they are allready generated
            pca: apply PCA or not
            threshold: Rij=heaviside(threshold-Sab) where Rij is the recurrence matrix
        """
        
        self.vid1Name=vid1Name
        self.vid2Name=vid2Name
        self.K=K
        self.windowsize=windowsize
        self.projectPath=projectPath
        self.skip=skip
        self.scale=scale
        self.pca=pca
        self.threshold=threshold
        
        
    def compute(self):
        data1,data2,hoghof=self.load_process_data()
        
      
        hoghof,ncomponent=self.applypca(hoghof)
            
        kmean=self.k_means_predict(hoghof,self.K)
        h2=self.histograms(data2,ncomponent,self.K,kmean)
        print("########### h2 #########################")
        print(h2)
        print("########## done h2 ###################")
        
        print("########### h1 #########################")
        h1=self.histograms(data1,ncomponent,self.K,kmean)
        print(h1)
        print("########## done h1 ###################")
        
        print("########### recurrence matrix is:   ################ ")
        self.recurrence_matrix(h1,h2,self.threshold)
        
       
        

       
       
        

            
        
            
        
        
        
        
        
    def applypca(self,data,ncp=None):
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
                var= sum(expvar[:i])
                ncomponent=i
                break
        
        if ncp:
            ncomponent=ncp
            
        print("######## Done applying PCA###############")
        print("Number of pc is:  "+str(ncomponent))
        
    
        return X[:,0:ncomponent],ncomponent 
    
    
    def histograms(self,data,ncomponent,K,kmean):
        """
        -input:
            -data: hoghof descriptor
            -number of principal component (comes from applying pca) 
            -K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
        """
        i=0
        gbp=data
        predclust=kmean.predict(self.applypca(preprocessing.scale((data.drop(['frame'],axis=1))),ncomponent)[0])
        gbp['pred']=predclust
        gbp=gbp.groupby('frame')
        train=np.zeros((len(gbp),K))
        for name,group in gbp:
           train[i][group.pred.values]=1
           #print(train[i])
           i+=1
           
        print("######### Generating histograms ###############")
         
        return train
    
    
    def k_means_predict(self,data,K,ini='k-means++'):
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
    
    
    def my_kernel(self,X,Y):
    
        return X.dot(Y.T)
    
    
    def OneSVM_predict(self,h,my_kernel):
        t0 = time()
        onesvm = svm.OneClassSVM( kernel=my_kernel)
        onesvm.fit(h)
        t=time()
        print("training time for svm----------: "+str(t-t0))
        return onesvm
    
    
    
    def SAB(self,h1,h2):
        svm1=self.OneSVM_predict(h1,self.my_kernel)
        svm2=self.OneSVM_predict(h2,self.my_kernel)
        Sab1=svm1.decision_function(h2)
        Sab2=svm2.decision_function(h1)
        return Sab1,Sab2
    
    
    
    def recurrence_matrix(self,h1,h2,threshold):
        sab1,sab2=self.SAB(h1,h2)
        Rij=np.zeros((h1.shape[0],h2.shape[0]))
        print((h1.shape[0],h2.shape[0]))
        print(sab1.shape)
        print(sab2.shape)
        
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                print((sab1[j]+sab2[i]))
                Rij[i][j]=np.where( (threshold-(sab1[j]+sab2[i]))>0,1,0 )
            
        print(Rij)
    
        return Rij




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
        data1=pd.read_csv(self.projectPath+"\\Automatic-measure-beta\\Data\\vid-gen1\\"+self.vid1Name+".txt",header=None, sep=r"\s+",skiprows=3)
        data2=pd.read_csv(self.projectPath+"\\Automatic-measure-beta\\Data\\vid-gen2\\"+self.vid2Name+".txt",header=None, sep=r"\s+",skiprows=3)
        
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
    
    
 
    
    
    

     im=Imitation("walk-complex","walk-simple",256,'C:\\Wail\\automatic-measure-of-imitation',skip=1,threshold=10)
     im.compute()      
     
        
        
        
        
        
        
        
        
        
        
    