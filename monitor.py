
from time import sleep
from pathlib import Path
from os.path import getctime
from os import unlink
from utils import binOpen
#from decompLib import Decomp
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

camPath='i3t'
figshape=[0,0]



trainPath='train/'+camPath
debugPath='debug/'+camPath
dicioPath='dicio/'+camPath



class NewfileHandler(FileSystemEventHandler):
    file_list=[]

    def on_created(self, event): # when file is created
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
        
        img,figshape=binOpen(fileName)


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

observer.start()
#process=Decomp(figshape,dicioPath)
# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()