
from time import sleep,time
from pathlib import Path
from os.path import getctime
from os import unlink
from util import binOpen,normalize
from denoiser import proxl1,ffd
from decomp import Decomp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import threading,os,shutil
import cv2
import numpy as np
from PIL import Image

#camPath='pan'
#figshape=(1536,2048)
#scale=0.5

camPath='i3t'
figshape=(480,640)
scale=1

train_limit=70
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


trainPath='train/'+camPath
debugPath='debug/'+camPath
dicioPath='dicio/'+camPath

def keyWatchdog():
    global key
    while key==None:
        key=input()
    print('Read '+key)


class NewfileHandler(FileSystemEventHandler):
    
    def on_created(self, event): # when file is created
        global train
        global run
        global train_copied
        global th
        file_list=[]

        # do something, eg. call your function to process the image
        fileName=str(event.src_path).split('.')
        fileName=fileName[0]+'.bin'
        print("Got created event for file "+fileName) #identifica o arquivo que gerou a interrupção
        fileOpen=False

        while not fileOpen: #espera o arquivo ser completamente copiado no diretorio monitorado
            try:
                f=open(fileName,'rb')
                fileOpen=True
                f.close()
            except:
                pass


        if run: #se o processamento está ativado
            t=time() 
            img,_= binOpen(fileName) #decodifica o arquivo .bin
            t2=time()
            print('Arquivo aberto ',str(t2-t))
            b_proj,a_proj=process.fit_proj(img) #decompoe pelo metodo de projecao
            print('Projection: '+str(process.fit_timer))

            limiar=0 #seta o limiar para considerar a imagem anomala

            en=np.linalg.norm(a_proj) #calcula a energia da anomalia
            if en>limiar:
                shutil.copy(fileName,'anom/'+fileName) #salva a imagem bin anomala na pasta anom


            #b_pnp,a_pnp=process.fit_pnp(img,proxl1,lamb2=0.1) #decompoe stoc
            #print('Stoc: '+str(process.fit_timer))
            #b_ffd,a_ffd=process.fit_pnp(img,ffd,max_iter=10) #decompoe ffd
            #print('Stoc: '+str(process.fit_timer))
            vis1 = np.concatenate((process.rescale(img),b_proj,a_proj), axis=1) #concatena as imagens originais, bg e anomalia
            #vis2= np.concatenate((process.rescale(img),b_pnp,a_pnp), axis=1)
            #vis3= np.concatenate((process.rescale(img),b_ffd,a_ffd), axis=1)
            #vis = np.concatenate((vis1,vis2), axis=0)
            vis= np.around(normalize(vis1,0,255)) #normaliza para exibicao

            im = Image.fromarray(vis)
            im=im.convert("L")
            im.save('out/'+fileName+'.jpeg') #salva a imagem como jpg para visualizacao


        if train: #se o modo de construir dataset de treino esta ativo
            destFile=fileName.replace('\\','/')
            destFile=destFile.split('/')
            destFile=trainPath+'/'+destFile[1]
            print('copying file',fileName,' to ',destFile) #salva os arquivos bin na pasta train
            shutil.copy(fileName,destFile)
            train_copied=train_copied+1
            print('Train images: ',train_copied)
            if train_copied>=train_limit: #se atingiu o tamanho do dataset
                print ('Finished train acquisition')
                th = threading.Thread(target=keyWatchdog) #reinicia monitor do teclado
                th.start()
                train=False #sai do modo de treino


        try: #cuida para manter a pasta de imagens com 30 arquivos no maximo
            file_list=sorted(Path(camPath).iterdir(),key=getctime)
            while len(file_list)>=30:
                unlink(file_list[0])
                #print('Deleted '+str(file_list[0]))
                file_list=sorted(Path(camPath).iterdir(),key=getctime)
        except:
            pass
'''   
    def on_modified(self, event): # when file is created
        # do something, eg. call your function to process the image
        print("Got modified event for file"+str(event.src_path))
        file_list=sorted(Path(folderPath).iterdir(),key=getctime)
        while len(file_list)>=30:
            unlink(file_list[0])
            print('Deleted '+str(file_list[0]))
            file_list=sorted(Path(folderPath).iterdir(),key=getctime)
'''

observer = Observer()
event_handler = NewfileHandler() # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path=camPath)
#instanciar um objeto da classe Decomp

key=None
th = threading.Thread(target=keyWatchdog)
th.start()

observer.start()
run=False
train=False
train_copied=0
# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    print('T - capture train images; S - Train dictionary and start;L - Load dictionary and start; P - Pause process; R - Resume process;')
    while True:
        if key is not None:      
            if key=='T' or key=='t':
                print('copying files from',camPath,' to ',trainPath)
                for filename in os.listdir(trainPath):
                    os.remove(os.path.join(trainPath,filename))
                train_copied=0
                train=True
                th.join()
                key=None
            elif key=='S' or key=='s':
                th.join()
                print('Starting process')
                process=Decomp(camPath,figshape,build=True,train_path=trainPath,scaling_factor=scale)
                run=True
                key=None
                th = threading.Thread(target=keyWatchdog)
                th.start()
            elif key=='P' or key=='p':
                th.join()
                print('Pausing process')
                run=False
                key=None
                th = threading.Thread(target=keyWatchdog)
                th.start()
            elif key=='R' or key=='r':
                th.join()
                print('Resume process')
                run=True
                key=None
                th = threading.Thread(target=keyWatchdog)
                th.start()
            elif key=='L' or key=='l':
                th.join()
                print('Load dic')
                process=Decomp(camPath,figshape,build=False,dicio_file=dicioPath,scaling_factor=scale)
                run=True
                key=None
                th = threading.Thread(target=keyWatchdog)
                th.start()
            elif key=='Q' or key=='q':
                raise KeyboardInterrupt
            else:
                th.join()
                key=None
                th = threading.Thread(target=keyWatchdog)
                th.start()
        
        sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()