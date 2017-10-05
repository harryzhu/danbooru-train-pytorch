import os
import glob
import math
import hashlib
import numpy as np
from faced import config

global conf
conf = config.conf

def get_np_cache(k):
    global conf
    npy_file = os.path.join(conf.get("cache",'cache_file_dir'),k + ".npy")
    if os.path.exists(npy_file):
        face_d = np.load(npy_file)
        #print("get npy from cache.")
        return face_d
    return None

def set_np_cache(k,v):
    npy_file = os.path.join(conf.get("cache",'cache_file_dir'),k + ".npy")
    np.save(npy_file,v)             

def parsePath(fullpath):
    if fullpath is None:
        return False
   
    last_dir_index = None
    if os.path.isfile(fullpath):
        last_dir_index = -2
    if os.path.isdir(fullpath):
        last_dir_index = -1

    if last_dir_index is None:
        return False

    list_path_seg = str.split(fullpath,"/")
    if list_path_seg[last_dir_index] is None:
        return False
    
    dict_return= {}
    dict_return["fullpath"] = fullpath
    dict_return["last_dir_name"] = list_path_seg[last_dir_index]

    fdir,fname = os.path.split(fullpath)
    dict_return["fulldir"] = fdir
    dict_return["file"] = fname

    fname_name, fname_ext = os.path.splitext(fname)
    dict_return["file_name"] = fname_name
    dict_return["file_ext"] = fname_ext


    return dict_return


