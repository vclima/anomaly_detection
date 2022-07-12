import os
import shutil
import time

temp_string='temp_'

def main(src='cams1/i3t',dest='i3t',delta_t=5):
    print('simulating input stream...')
    print('copying files from',src,' to ',dest)

    # wipe destination directory
    for filename in os.listdir(dest):
        os.remove(os.path.join(dest,filename))
        
    for k,filename in enumerate(os.listdir(src)):
        # create temp filename
        temp_filename=filename.split('.')
        temp_filename=temp_filename[0]+'.tmp'
        print(temp_filename)
        print(filename)
        
        # copy one file
        shutil.copy(os.path.join(src,filename),os.path.join(dest,temp_filename))
        print('copied file number ',k)
           
        # rename copied filed
        os.rename(os.path.join(dest,temp_filename),os.path.join(dest,filename))
        
        # sleep delta_t seconds
        time.sleep(delta_t)
     
     
if __name__ == "__main__":
    main()