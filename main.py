
from time import sleep
from pathlib import Path
from os.path import getctime
from os import unlink
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

folderPath='ftp'

class NewfileHandler(FileSystemEventHandler):
    file_list=[]

    def on_created(self, event): # when file is created
        # do something, eg. call your function to process the image
        print("Got created event for file"+str(event.src_path))
        file_list=sorted(Path(folderPath).iterdir(),key=getctime)
        while len(file_list)>=30:
            unlink(file_list[0])
            print('Deleted '+str(file_list[0]))
            file_list=sorted(Path(folderPath).iterdir(),key=getctime)
    
    def on_modified(self, event): # when file is created
        # do something, eg. call your function to process the image
        print("Got modified event for file"+str(event.src_path))
        file_list=sorted(Path(folderPath).iterdir(),key=getctime)
        while len(file_list)>=30:
            unlink(file_list[0])
            print('Deleted '+str(file_list[0]))
            file_list=sorted(Path(folderPath).iterdir(),key=getctime)

observer = Observer()
event_handler = NewfileHandler(ignore_directories=True) # create event handler
# set observer to use created handler in directory
observer.schedule(event_handler, path=folderPath)
observer.start()

# sleep until keyboard interrupt, then stop + rejoin the observer
try:
    while True:
        sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()