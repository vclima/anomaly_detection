import cv2
from PIL import Image
import numpy as np
import pathlib 
from os.path import getctime,isfile,isdir,join
from os import listdir
import rpca
from scipy.sparse.linalg import svds
from datetime import datetime


class Decomp:
    '''
    Decomp Class
    pars:
        name             : camera name. either 'i3t', 'uvc', 'pan' or 'rgb'
        fig_shape        : expected input size before scaling. stored as (n1,n2) 
        scalining_factor : number in (0,1]. default value is 1
        build            : build dicio from scratch or not
        debug_dicio      : save background images from training set
        
    atts: 
        n1,n2      : expected input size before rescaling
        n          : n1*n2
        sn1,sn2    : expected input size after rescaling
        sn         : sn1*sn2
        
        dicio : RPCA-based dictionary that describes the background
                
        BGBuildPeriod: Time elapsed during BG analysis in hours
        BuildRate: Time between BG observations in minutes
        BG_Images: BG observations

     Methods:
        rescale()
        build_dicio()
        load_dicio()
        fit_proj()
        fit_pnp()
'''

    def __init__(self,name,fig_shape,scaling_factor=1,build=True,dicio_file='dicio',train_path=None,debug_dicio=False):
        # camera name 
        self.name=name
    
        # input atts
        self.n1,self.n2=fig_shape
        self.n=self.n1*self.n2
        
        # scaled input atts
        self.scaling_factor=scaling_factor
        self.sn1,self.sn2=int(round(scaling_factor*self.n1)),int(round(scaling_factor*self.n2))
        self.sn=self.sn1*self.sn2

        # dicio atts
        self.dicio=None
        self.dicio_file=dicio_file
        self.train_path=train_path
        
        # fit atts
        
	
        if build==True and train_path is None:
            print('Missing train_path')
            return
	    
        if dicio_file is None:
            print('Missing dicio_file')

        if build==True:
            self.build_dicio(debug=debug_dicio)
        else:
            self.load_dicio()


    def rescale(self,frame):
        frame = cv2.resize(frame, None, fx=self.scaling_factor, fy=self.scaling_factor, interpolation=cv2.INTER_AREA)
        #frame=cv2.normalize(frame, None, 1.0,0.0, cv2.NORM_MINMAX,cv2.CV_32F)
        return frame

    def build_dicio(self,tol=1e-3,debug=False,debug_path='debug'):
        # open training images
        files = [join(self.train_path,f) for f in listdir(self.train_path) if isfile(join(self.train_path, f))]
        frames=[]
        for filename in files:
            # open file
            frame=cv2.imread(filename)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame=np.array(frame)

            # rescale 
            frame=self.rescale(frame)

            # append
            frames.append(frame.reshape(self.sn))
            
        # stack training images
        Y=np.array(frames).transpose()
        
	# create dicio
        B,_,_,r=rpca.pcp(Y,tol=tol)
        r=np.linalg.matrix_rank(B)
        U,sigma,_=svds(B,k=r)
        print(r)

	# default pars
        self.lambda1 = 1.0/np.sqrt(self.sn)/np.mean(sigma) 
        self.lambda2 = 1.0/np.sqrt(self.sn)  

	# current date
        date = datetime.now().strftime("%Y_%m_%d-%I%M%S_%p")

	# save background from training set
        if debug==True:
            fname=join(debug_path,self.name)+date+'.npy'
            with open(fname, 'wb') as f:
                np.save(f, B)

        # save dicio
        if self.dicio_file is None:
            self.dicio_file=join('dicio',self.name)+date+'.npy'

        with open(self.dicio_file, 'wb') as f:
            np.save(f, U)

        self.dicio=U
        return

    def load_dicio(self):
        if isdir(self.dicio_file):
            file_list=sorted(pathlib.Path(self.dicio_file).iterdir(),key=getctime)
            try:
                self.dicio=np.load(file_list[-1])
                print('Loaded dictionary from '+str(file_list[-1]))
                return
            except:
                print('Cannot load dictionary from '+str(file_list[-1]))
                self.dicio=None
                return
        elif isfile(self.dicio_file):
            try:
                self.dicio=np.load(self.dicio_file)
                print('Loaded dictionary from '+str(self.dicio_file))
                return
            except:
                print('Cannot load dictionary from '+str(file_list[-1]))
                self.dicio=None
                return
        else:
            print('Cannot load dictionary from '+str(self.dicio_file))
            self.dicio=None	
            return

    def fit_proj(self,frame):
        # rescale input
        frame=self.rescale(frame)
        
        # vectorize input 
        frame_vec=np.array(frame).reshape((self.sn,1))
        print(frame.shape)
        print(self.dicio.shape)
        print(frame_vec.shape)
        
        # proj
        alpha=np.matmul(self.dicio.T,frame_vec)
        b=np.matmul(self.dicio,alpha)
        a=frame_vec-b
        
        # reshape back
        b=b.reshape((self.sn1,self.sn2))
        a=a.reshape((self.sn1,self.sn2))
        #B=cv2.normalize(B, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        #A=cv2.normalize(A, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # 
        #A=util.normalize(A)
        #B=util.normalize(B)
        return b,a

    def fit_pnp(self,Im,tol=1e-7):
        #L,S -> B,A (ATENÇÃO PARA B,A JÁ EXISTENTES)	
        m,r=self.BGDic.shape
        #A = np.zeros((r, r))
        #B = np.zeros((m, r))

        
        ImArray=np.array(Im).reshape((self.Width*self.Length,1))
        si,ai=solve_proj(ImArray[:,0],self.BGDic,self.lambda1,self.lambda2,tol=tol)

	#A = A + np.outer(si, si)
	#B = B + np.outer(ImArray[:,0] - ai, si)
        
        b_frame=np.array(self.BGDic.dot(si).reshape(self.Length,self.Width))
        a_frame=np.array(ai.reshape(self.Length,self.Width))

        S=cv2.normalize(a_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        L=cv2.normalize(b_frame, None, 1,0, cv2.NORM_MINMAX, cv2.CV_32F)
        return L,S

    
