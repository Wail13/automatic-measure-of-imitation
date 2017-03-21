# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:04:13 2017

@author: isir
"""

    
class ImitationM:
    
    from sklearn import svm
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import KFold
    from sklearn.cluster import KMeans
    from sklearn import preprocessing
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
    
    def __init__(self,vid1Name,vid2Name,K,projectPath,skip=1,scale=1,threshold=0.1):
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
        self.projectPath=projectPath
        self.skip=skip
        self.scale=scale
        self.threshold=threshold
        
        
        
    def compute(self):
        
        data1,data2,hoghof=self.load_process_data()
        kmean=self.k_means_predict(hoghof,self.K)
        h2=self.histograms(data2,self.K,kmean)
        h1=self.histograms(data1,self.K,kmean)
            
        return self.recurrence_matrix(h1,h2,self.threshold)
    
    
    
    def Cross_validation(self,h):
     n_folds=4
     nu=np.linspace(0.001, 1, 1000)
     results=[]
     for d in nu:
         onesvm = svm.OneClassSVM( kernel=my_kernel, nu=d)
         hypothesisresults=[]
         for train, test in KFold(len(h), n_folds):
             onesvm.fit(h[train]) # fit
             hypothesisresults.append(np.mean(onesvm.predict(h[test])==1))
            
         results.append(np.mean(hypothesisresults))
        #print(results)
        
        
     return nu[np.argmax(results)]
        
   



    def histograms(self,data,K,kmean):
        """
        -input:
            -data: hoghof descriptor
            -number of principal component (comes from applying pca) 
            -K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
        """
        i=0
        gbp=data
        predclust=kmean.predict(preprocessing.scale(data.drop(['frame'],axis=1)))
        gbp['pred']=predclust
        gbp=gbp.groupby('frame')
        train=np.zeros((len(gbp),K))
        for name,group in gbp:
           train[i][group.pred.values]=1
           #print(train[i])
           i+=1
           
#        print("######### Generating histograms ###############")
         
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
        nuu=self.Cross_validation(h)
        print('nu coeficient is:    '+str(nuu))
        onesvm = svm.OneClassSVM( kernel=my_kernel,nu=nuu)
        onesvm.fit(h)
        t=time()
#        print("training time for svm----------: "+str(t-t0))
        return onesvm
    
    
    
    def SAB(self,h1,h2):
        svm1=self.OneSVM_predict(h1,self.my_kernel)
        svm2=self.OneSVM_predict(h2,self.my_kernel)
        Sab1=(svm1.decision_function(h2)-svm1.intercept_)
        Sab2=(svm2.decision_function(h1)-svm1.intercept_)

        return Sab1,Sab2
    
    
    
    def recurrence_matrix(self,h1,h2,threshold):
        """
        input:
            -h1,h2 histograms 
        output:
            -Rij:Recurrence matrix
            -Dij:Raw Recurrence matrix (before applying the threshold)
        """
        sab1,sab2=self.SAB(h1,h2)
        Rij=np.zeros((h1.shape[0],h2.shape[0]))
        Dij=np.zeros((h1.shape[0],h2.shape[0]))
      
        for i in range(h1.shape[0]):
            for j in range(h2.shape[0]):
                Rij[i][j]=np.where(np.square(sab1[j]-sab2[i])-threshold <0, 1, 0 )
                Dij[i][j]= np.square(sab1[j]-sab2[i])
                
    
        return Rij,Dij




    def load_process_data(self):
        """
        output: 
            -data1: dataframe for the first videoc 'STIP' points
            -data2: dataframe for the second video 'STIP' points
            -hoghof: descriptor of both videos. We used a combination of  histogram of gradients and histogram of flow 
        """
        
#        print("################ loading data ######################")
        
        pathVid1="\\Automatic-measure-beta\\Data\\"+self.vid1Name+".avi"
        pathVid2="\\Automatic-measure-beta\\Data\\"+self.vid2Name+".avi"


        # Generate a file .txt that contains STIP points  
        if self.skip!=1:
            os.system(self.projectPath+ImitationM.pathSTIP+" -f "+self.projectPath+pathVid1+" -o "+self.projectPath+ImitationM.pathout1+self.vid1Name+".txt -vis no")
            os.system(self.projectPath+ImitationM.pathSTIP+" -f "+self.projectPath+pathVid2+" -o "+self.projectPath+ImitationM.pathout2+ self.vid2Name+".txt -vis no")
            
        
        #loading the STIP into a panda dataframe
        data1=pd.read_csv(self.projectPath+"\\Automatic-measure-beta\\Data\\vid-gen1\\"+self.vid1Name+".txt",header=None, sep=r"\s+",skiprows=3)
        data2=pd.read_csv(self.projectPath+"\\Automatic-measure-beta\\Data\\vid-gen2\\"+self.vid2Name+".txt",header=None, sep=r"\s+",skiprows=3)
        
        #drop useless columns
        data1.drop(data1.columns[[0,1,2,4,5,6]], axis=1, inplace=True)
        data2.drop(data2.columns[[0,1,2,4,5,6]], axis=1, inplace=True)
        
        #columns names
        column=["hgf"+str(i) for i in range(162)]
        column.insert(0,'frame')
        #naming the columns
        data1.columns=column
        data2.columns=column
        
#        # Working in the same frame range
        data1=data1[(data1.frame>=max(min(data1.frame),min(data2.frame))) & (data1.frame<=min(max(data1.frame),max(data2.frame)))]
        data2=data2[(data2.frame>=max(min(data1.frame),min(data2.frame))) & (data2.frame<=min(max(data1.frame),max(data2.frame)))]
        
        #merging the two dataframes
        data=data1.append(data2,ignore_index=True)
        data=data.drop(['frame'],axis=1)
#        data=preprocessing.scale(data)
                        
        return data1,data2,data
    
    


#
#
im=ImitationM("walk-complex","walk-complex",7,'C:\\Wail\\automatic-measure-of-imitation',skip=1,threshold=0.0000000000001)
Rij,Dij=im.compute()  
ax=sns.heatmap(Rij, xticklabels=2, yticklabels=False)
ax.invert_yaxis()
ax

ax=sns.heatmap(np.where(Dij-0.09 <0, 1, 0 ) )

