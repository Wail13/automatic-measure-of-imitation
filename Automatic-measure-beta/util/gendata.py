# -*- coding: utf-8 -*-
"""
Created on Wed Mar 08 11:32:52 2017

@author: isir
""" 


import sys, os
#project path input args
#pathproject="C:\\Wail"

pathSTIP="\\Automatic-measure-beta\\util\\cvml10-actions\\cvml10-actions\\stip\\bin\\stipdet.exe "




#video to parse name params ---arg---
#vidname1="daria_wave1"
#vidname2="denis_wave1"


#video input path

# Write folder path
pathout1="\\Automatic-measure-beta\\Data\\vid-gen1\\"
pathout2="\\Automatic-measure-beta\\Data\\vid-gen2\\"





def main():
    listarg=[] 
    for arg in sys.argv[1:4]:
        listarg.append(arg)
        
        
    pathproject=listarg[0]
    vidname1=listarg[1]
    vidname2=listarg[2]
    pathVid1="\\Automatic-measure-beta\\Data\\"+vidname1+".avi"
    pathVid2="\\Automatic-measure-beta\\Data\\"+vidname2+".avi"


    os.system(pathproject+pathSTIP+" -f "+pathproject+pathVid1+" -o "+pathproject+pathout1+vidname1+".txt -vis no")
    os.system(pathproject+pathSTIP+" -f "+pathproject+pathVid2+" -o "+pathproject+pathout2+vidname2+".txt -vis no")

       
        
    
if __name__ == "__main__":
    main()
    
    
  



    


