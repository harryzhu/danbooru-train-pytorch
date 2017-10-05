import time
t_start = time.time()
import os
from multiprocessing import cpu_count
from multiprocessing import Pool, Lock, Value
from multiprocessing.dummy import Pool as ThreadPool
import logging
from faced.core import *

fd = Facedetect("conf/conf.toml")

lock = Lock()
mp_counter = Value('i',0)

print(fd.get_face_recognition("images/recognitionImages/027_025.jpg"))
#print(fd.scanFaceRecognitionDir())

def run(f):
    global lock, mp_counter
    with lock:
        mp_counter.value += 1
    print('NO. (%d/%d) task starts. PID: %d ' %(mp_counter.value, len(files), os.getpid()))
    fd.save_faces(f)
    return
    
def MultiProcessing_generateFacesFromSouceDir(n=4):
    pool = Pool(n)
    rl = pool.map(run, files)
    
    pool.close()
    pool.join()
    #print(rl)

def MultiThreading_generateFacesFromSouceDir(n=6):
    tpool = ThreadPool(n)
    trl = tpool.map(run,files)
    tpool.close()
    tpool.join()

def generateFacesFromSouceDir():
    for f in files:
        fd.save_faces(f)

ccount = cpu_count()
if ccount >= 8:
    n = ccount * 2 - 2
    print("CPU count:",ccount," will use multi-process to run the job.")
    #MultiProcessing_generateFacesFromSouceDir(n)
elif ccount < 8 and ccount > 1:
    n = ccount * 2 - 2
    print("CPU count:",ccount," will use multi-threading to run the job.")
    #MultiThreading_generateFacesFromSouceDir(n)
else:
    print("CPU count:",ccount," will run the job directly.")
    #generateFacesFromSouceDir()

t_stop = time.time()
msg = "TOTAL running time: "+ str(t_stop - t_start) + " seconds/"  + " images"
print(msg)
#logging.info(msg)