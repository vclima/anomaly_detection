import cv2
import numpy as np
from pathlib import Path
from os.path import getctime,isfile,isdir
from rpca import pcp,solve_proj
from scipy.sparse.linalg import svds
from datetime import datetime



class decomp:

    '''Class: Background 

    Attributes:
        BGBuildPeriod: Time elapsed during BG analysis in hours
        BuildRate: Time between BG observations in minutes
        Width: Frame Width
        Length: Frame Length
        BG_Images: BG observations
        BGDic: RPCA-based dictionary that describes the background

     Methods:
        BuildDictionary(tol): Create the RPCA-based dictionary with tol tolerance '''

    def __init__(self,dictPath='dict',scalingFactor=1,dictBuild=False):

        self.Width=int(640*scalingFactor)
        self.Length=int(480*scalingFactor)

        self.BGDic=None

        if dictBuild:
            self.BuildDictionary(dictPath)
        else:
            self.LoadDictionary(dictPath)

   

    def BuildDictionary(self,img_path,tol=1e-3):

        L,_,_,r=pcp(self.BG_Images,tol=tol)
        r=np.linalg.matrix_rank(L)
        self.BG_Images=None

        U,sigma,_=svds(L,k=r)
        m=self.Width*self.Length

        self.lambda1 = 1.0/np.sqrt(m)/np.mean(sigma) 
        self.lambda2 = 1.0/np.sqrt(m) # 0.05 

        date = datetime.now().strftime("%Y_%m_%d-%I%M%S_%p")
        fname='dict/L_'+date+'.npy'
        with open(fname, 'wb') as f:
            np.save(f, L)
        fname='dict/dic_'+date+'.npy'
        with open(fname, 'wb') as f:
            np.save(f, U)

        self.BGDic=U
        return

    def LoadDictionary(self,dic):
        if isdir(dic):
            file_list=sorted(Path(dic).iterdir(),key=getctime)
            try:
                U=np.load(file_list[-1])
                print('Loaded dictionary from '+str(file_list[-1]))
                self.BGDic=U
                return
            except:
                print('Cannot load dictionary from '+str(file_list[-1]))
                self.BGDic=None
                return
        elif isfile(dic):
            try:
                U=np.load(file_list[-1])
                self.BGDic=U
                print('Loaded dictionary from '+str(file_list[-1]))
                return
            except:
                print('Cannot load dictionary from '+str(file_list[-1]))
                self.BGDic=None
                return
        else:
            print('Cannot load dictionary from '+str(dic))
            self.BGDic=None
            return

    def DecomposeProj(self,Im):
        #L,S -> B,A
        ImArray=np.array(Im).reshape((self.Width*self.Length,1))
        alpha=np.matmul(self.BGDic.T,ImArray)
        L=np.matmul(self.BGDic,alpha)
        S=ImArray-L
        L=L.reshape((self.Length,self.Width))
        S=S.reshape((self.Length,self.Width))
        L=cv2.normalize(L, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        S=cv2.normalize(S, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        return L,S

    def DecomposeStoc(self,Im,tol=1e-7):
        #L,S -> B,A (ATENÇÃO PARA B,A JÁ EXISTENTES)
        m,r=self.BGDic.shape
        A = np.zeros((r, r))
        B = np.zeros((m, r))

        
        ImArray=np.array(Im).reshape((self.Width*self.Length,1))
        si,ai=solve_proj(ImArray[:,0],self.BGDic,self.lambda1,self.lambda2,tol=tol)

        A = A + np.outer(si, si)
        B = B + np.outer(ImArray[:,0] - ai, si)
        
        b_frame=np.array(self.BGDic.dot(si).reshape(self.Length,self.Width))
        a_frame=np.array(ai.reshape(self.Length,self.Width))

        S=cv2.normalize(a_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        L=cv2.normalize(b_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        return L,S

    def DecomposePCP(self,Im,tol=1e-7):

        return
    
    