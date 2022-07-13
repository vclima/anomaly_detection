import pathlib 
from os.path import getctime,isfile,isdir,join
from os import listdir
from datetime import datetime
from timeit import default_timer as timer

from scipy.sparse.linalg import svds
import cv2
import numpy as np
import util


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

    def __init__(self,name,fig_shape,scaling_factor=1,build=True,dicio_file=None,train_path=None,debug_dicio=False):
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
        self.r=-1
        
        # fit atts
        self.fit_timer=-1
        self.lamb1=-1
        self.lamb2=-1
        self.pnp_admm_cte=-1
        
	
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
        return np.array(frame)

    def build_dicio(self,tol=1e-1,debug=False,debug_path='debug'):
        # open training images
        files = [join(self.train_path,f) for f in listdir(self.train_path) if isfile(join(self.train_path, f))]
        frames=[]
        for k,filename in enumerate(files):
            # open file
            frame,_=util.binOpen(filename)
            
            # debug
            # print('Read image',k,'/',len(files))

            # rescale 
            frame=self.rescale(frame)

            # append
            frames.append(frame.reshape(self.sn))
            
        # stack training images
        Y=np.array(frames).transpose()
        
	# create dicio
        B,_,_,rank=util.pcp(Y,tol=tol)
        #self.r=np.linalg.matrix_rank(B)
        self.r=rank
        
        # prevent svds from crashing
        if self.r<=0:
            self.r=1
        elif self.r>=min(B.shape):
            self.r=min(B.shape)-1
        
        # run svds
        U,sigma,_=svds(B,k=self.r)
        
        self.dicio=U
        print('Dictionary built with',self.r,'atoms')

	# default pars and default pnp_admm_cte
        self.lamb1 = 1.0/np.sqrt(self.sn)/np.mean(sigma) 
        self.lamb2 = 1.0/np.sqrt(self.sn)  
        I=np.eye(self.r)
        self.pnp_admm_cte=np.linalg.inv(self.dicio.transpose().dot(self.dicio)+self.lamb1*I).dot(self.dicio.transpose()) 

	# current date
        date = datetime.now().strftime("%Y_%m_%d-%I%M%S_%p")

	# save background from training set
        if debug==True:
            fname=join(debug_path,self.name)+date+'.npz'
            with open(fname, 'wb') as f:
                np.save(f, B)
                print('Saving background images of the training set in',fname)

        # check if dicio_name is set. if not, use default name
        if self.dicio_file is None:
            self.dicio_file=join('dicio',self.name)+'/'+date+'.npz'
    
        print(date)
        print(self.name)
        print(self.dicio_file)

        # save dicio file with pars
        with open(self.dicio_file, 'wb') as f:
            np.savez(f,dicio=self.dicio,lamb1=self.lamb1,lamb2=self.lamb2,pnp_admm_cte=self.pnp_admm_cte)
            print('Saving dictionary and default parameters in',self.dicio_file)


    def load_dicio(self):
        if isdir(self.dicio_file):
            file_list=sorted(pathlib.Path(self.dicio_file).iterdir(),key=getctime)
            try:
                var=np.load(file_list[-1])
                self.dicio=var['dicio']
                self.lamb1=var['lamb1']
                self.lamb2=var['lamb2']
                self.r=self.dicio.shape[1]
                self.pnp_admm_cte=var['pnp_admm_cte']
                print('Loaded dictionary from '+str(file_list[-1]))
                return
            except:
                print('Cannot load dictionary from'+str(file_list[-1]))
                self.dicio=None
                return
        elif isfile(self.dicio_file):
            try:
                var=np.load(self.dicio_file)
                self.dicio=var['dicio']
                self.lamb1=var['lamb1']
                self.lamb2=var['lamb2']
                self.r=self.dicio.shape[1]
                self.pnp_admm_cte=var['pnp_admm_cte']
                print('Loaded dictionary from '+str(self.dicio_file))
                return
            except:
                print('Cannot load dictionary from'+str(file_list[-1]))
                self.dicio=None
                self.lamb1=-1
                self.lamb2=-1
                self.pnp_admm_cte=-1
                return
        else:
            print('Cannot load dictionary from '+str(self.dicio_file))
            self.dicio=None	
            self.lamb1=-1
            self.lamb2=-1
            self.pnp_admm_cte=-1
            return

    def fit_proj(self,frame):
        # start timer
        start_time=timer()
    	
        # rescale input
        frame=self.rescale(frame)
        
        # vectorize input 
        frame_vec=frame.reshape((self.sn,1))
        
        # decompose
        alpha=np.matmul(self.dicio.T,frame_vec)
        b=np.matmul(self.dicio,alpha)
        a=frame_vec-b
        
        # reshape back
        b=b.reshape((self.sn1,self.sn2))
        a=a.reshape((self.sn1,self.sn2))
 
 	# normalize in [0,1]
        #a=util.normalize(a)
        #b=util.normalize(b)
        
        # update timer
        end_time = timer()    
        self.fit_timer=end_time-start_time
        
        return b,a

    def fit_pnp(self,frame,den,denpar=None,mu=1,lamb1=None,lamb2=None,tol=1e-7,max_iter=100,debug=False):
        # start timer
        start_time=timer()
    	
        # rescale input
        frame=self.rescale(frame)
        
        # vectorize input 
        frame_vec=frame.reshape((self.sn,1))    
        
        # default regularization parameter 1 
        # lamb1 is also used in the computation of pnp_admm_cte. if the default lamb1 value is used, then cte is pre-computed 
        # in the dictionary building step, and it is stored with the other dictionary parameters. otherwise, we compute 
        # it again.  
        if lamb1 is None:
            lamb1=self.lamb1
            cte=self.pnp_admm_cte
        else:
            I=np.eye(self.r)
            cte=np.linalg.inv(self.dicio.transpose().dot(self.dicio)+lamb1*I).dot(self.dicio.transpose()) 
        
        # default regularization parameter 2 
        if lamb2 is None:
            lamb2=self.lamb2
 
        # default denoiser parameter
        if denpar is None:
            denpar=lamb2/mu    
              
        # ADMM solver for PnP decomposition 
        s=np.zeros((self.r,1))   # bg representation
        a=np.zeros((self.sn,1))  # anomaly component
        e=np.zeros((self.sn,1))  # dummy of the anomaly component
        m=np.zeros((self.sn,1))  # dual variable 
        
        # begin loop
        err=tol+1
        ite=0
        while (err>tol) and (ite<max_iter):
            ite=ite+1
            skm1=s
            akm1=a
        
            # update s
            s=cte.dot(frame_vec-a)
 
            # update a 
            Ds=self.dicio.dot(s)
            #ak=self.f(yk-Dsk,self.fpars) altmin
            a=(1/(1+mu))*(frame_vec-Ds+mu*e+m)
          
            # update dual variable m
            m=m+mu*(e-a)
    
            # PnP update dummy variable e 
            a_devec=a.reshape((self.sn1,self.sn2))
            m_devec=m.reshape((self.sn1,self.sn2))
            e_devec=den(a_devec - m_devec/mu, denpar)
            e=e_devec.reshape((self.sn,1))
            
            
            ## calculate reports
            # err
            err=np.linalg.norm(frame_vec-Ds-a,ord=2)
            
            # debug
            if debug==True: #and ite%self.iter_print==0:
                print(' ite=',ite,'. err=',err)
               
                
        # background component 
        b=self.dicio.dot(s)

        # reshape back
        b=b.reshape((self.sn1,self.sn2))
        a=a.reshape((self.sn1,self.sn2))
       
	# normalize in [0,1]
        #a=util.normalize(a)
        #b=util.normalize(b)
        
        # update timer
        end_time = timer()    
        self.fit_timer=end_time-start_time
        
        return b,a

    
