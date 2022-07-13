
from time import sleep
from pathlib import Path
from os.path import getctime
from os import unlink
from util import binOpen
from decomp import Decomp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import threading,os,shutil

camPath='i3t'
figshape=(640,480)
train_limit=70



trainPath='train/'+camPath
debugPath='debug/'+camPath
dicioPath='dicio/'+camPath

def keyWatchdog():
    global key
    print('keyboard thread online')
    key=input()


class NewfileHandler(FileSystemEventHandler):
    
    def on_created(self, event): # when file is created
        global train
        global run
        file_list=[]

        # do something, eg. call your function to process the image
        fileName=str(event.src_path).split('.')
        fileName=fileName[0]+'.bin'
        print("Got created event for file "+fileName)
        fileOpen=False

        while not fileOpen:
            try:
                f=open(fileName,'rb')
                fileOpen=True
                f.close()
            except:
                pass
        if run:
            img,_=binOpen(fileName)
            b_proj,a_proj=process.fit_proj(img)
            b_pnp,a_pnp=process.fit_pnp(img)
        if train:
            destFile=fileName.replace('\\','/')
            destFile=destFile.split('/')
            print('copying file',fileName,' to ',camPath+fileName[1])
            shutil.copy(fileName,camPath+fileName[1])
            train_copied=train_copied+1
            print('copied file number ',train_copied)
            if train_copied>=train_limit:
                print ('Finished train aquisition')
                th = threading.Thread(target=keyWatchdog)
                train=False


        try:
            file_list=sorted(Path(camPath).iterdir(),key=getctime)
            while len(file_list)>=30:
                unlink(file_list[0])
                print('Deleted '+str(file_list[0]))
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

th = threading.Thread(target=keyWatchdog)
th.start()

observer.start()
run=False
train=False
train_copied=0
key=None
# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    print('T - capture train images; B - begin process; S - stop process')
    while True:      
        if key=='T':
            print('copying files from',camPath,' to ',trainPath)
            key=None
            for filename in os.listdir(trainPath):
                os.remove(os.path.join(trainPath,filename))
            train_copied=0
            train=True
            th.join()
        if key=='B':
            key=None
            print('Starting process')
            process=Decomp(camPath,figshape)
            process.build_dicio()
            run=True
        if key=='S':
            key=None
            print('Stopping process')
            run=False
        sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()