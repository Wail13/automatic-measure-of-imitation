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
    
    
    def __init__(self,vid1Name,vid2Name,K,projectPath,windowsize=1,skip=1):
        """
        vid1Name: the name of the video located in the Data folder
        K: number of words in the codebook or the codewords or visual vocabulary or K means number of clusters
        projectPath: Where the project is located in your computer example "C:\\Wail"
        windowsize: number of frames for each training vector  example of 2 frame : vect[1,2]--vect[3,4]--vect[5,6]
        skip: skip generating STIP points if they are allready generated
        """
        
        self.vid1Name=vid1Name
        self.vid2Name=vid2Name
        self.K=K
        self.windowsize=windowsize
        self.projectPath=projectPath
        self.skip=skip
        
        
        
        
    def load_process_data(self):
        
        pathVid1="\\Automatic-measure-beta\\Data\\"+self.vid1Name+".avi"
        pathVid2="\\Automatic-measure-beta\\Data\\"+self.vid2Name+".avi"


        # Generate a file .txt that contains STIP points  
        if self.skip!=1:
            os.system(self.projectPath+Imitation.pathSTIP+" -f "+self.projectPath+pathVid1+" -o "+self.projectPath+Imitation.pathout1+self.vid1Name+".txt -vis no")
            os.system(self.projectPath+Imitation.pathSTIP+" -f "+self.projectPath+pathVid2+" -o "+self.projectPath+Imitation.pathout2+ self.vid2Name+".txt -vis no")
            
        
        
        data1=pd.read_csv('../Data/vid-gen1/'+self.vid1Name+'.txt',header=None, sep=r"\s+",skiprows=3)
        data2=pd.read_csv('../Data/vid-gen2/'+self.vid1Name+'.txt',header=None, sep=r"\s+",skiprows=3)
        
        
        
        
        
        
        
        
    