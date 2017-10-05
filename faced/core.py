import os
import time
import logging
import toml
import glob
import cv2
import dlib
import math
import hashlib 
import numpy as np

class Singleton(type):
    _inst = {}
    def __call__(self, *args, **kw):
        if self not in self._inst:
            self._inst[self] = super(Singleton, self).__call__(*args, **kw)
        return self._inst[self]

class Facedetect(metaclass = Singleton):
    def __init__(self,cfgFile):
        self._loadConfig(cfgFile)
        self._set_face_detector()
        self._set_landmark_predictor()
        self._set_face_recognition_model()
        self._set_candidates_list()
        self._set_candidates_face_desciptors()
            
    def _loadConfig(self,cfgFile):
        if os.path.exists(cfgFile) is None:
            return None

        self.conf = toml.load(cfgFile)
        if self.conf is None:
            logging.error('cannot load the config file: ' + str(cfgFile))
        else:
            logging.info('loaded the config file: ' + str(cfgFile))
    
    def getConfig(self,section,name):
        return self.conf[section][name]

    def getConfigInt(self,section,name):
        return int(self.conf[section][name])

    def getConfigFloat(self,section,name):
        return float(self.conf[section][name])

    def getConfigBool(self,section,name):
        return bool(self.conf[section][name])

    def _set_face_detector(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def _set_landmark_predictor(self):
        lp = self.getConfig('dataset','dataset_shape_predictor')
        if os.path.isfile(lp):
            self.landmark_predictor = dlib.shape_predictor(lp)
        else:
            print(lp + " is invalid, pls check your config file.")
            logging.error(lp + " is invalid, pls check your config file.")

    def _set_face_recognition_model(self):
        fr = self.getConfig('dataset','dataset_face_recognition')
        if os.path.isfile(fr):
            self.face_recognition_model = dlib.face_recognition_model_v1(fr)
        else:
            print(fr + " is invalid, pls check your config file.")
            logging.error(fr + " is invalid, pls check your config file.")

    def _set_candidates_list(self):
        candidatesDir = self.getConfig('path', 'candidates_dir')
        if candidatesDir is None:
            logging.error("candidatesDir cannot be empty")
            return False
        
        candidates = []
        for rt,dirs,files in os.walk(candidatesDir):
            if rt is None:
                return None

            for f in files:
                fname = os.path.splitext(f)
                fpath = os.path.join(candidatesDir,f)
                if fname[1] in ['.jpg','.png']:
                    candidates.append(f)
        self.candidates_list = candidates
        return True

    def _set_candidates_face_desciptors(self):
        if self.candidates_list is None:
            print("candidates_list should be a list")
            return False

        str_cl = ",".join(self.candidates_list)
        sha1 = hashlib.sha1()  
        sha1.update(str_cl.encode("utf8"))
        sha1_str_cl = sha1.hexdigest() 

        c_l = self.get_np_cache(sha1_str_cl)
        if c_l is not None:
            self.candidates_face_descriptors = c_l
            return True

        candidates_descriptors = []

        for c in self.candidates_list:
            f = os.path.join(self.getConfig('path','candidates_dir'), c)
            #print("Processing file: {} :{}".format(c,f))
            img_descriptor = self.get_face_descriptor(f)
            v = np.array(img_descriptor)
            candidates_descriptors.append(v)
        self.set_np_cache(sha1_str_cl,candidates_descriptors)
        self.candidates_face_descriptors = candidates_descriptors

    def get_face_detector(self):
        return self.face_detector

    def get_landmark_predictor(self):
        return self.landmark_predictor

    def get_face_recognition_model(self):
        return self.face_recognition_model

    
    def get_np_cache(self,k):
        npy_file = os.path.join(self.getConfig("cache",'cache_file_dir'),k + ".npy")
        if os.path.exists(npy_file):
            return np.load(npy_file)
        return None
    
    
    def set_np_cache(self,k,v):
        npy_file = os.path.join(self.getConfig("cache",'cache_file_dir'),k + ".npy")
        np.save(npy_file,v)             

    
    def parsePath(self,fullpath):
        if fullpath is None:
            return False

        fullpath = os.path.abspath(fullpath)
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
        dict_return["last_dir"] = list_path_seg[last_dir_index]

        fdir,fname = os.path.split(fullpath)
        dict_return["full_dir"] = fdir
        dict_return["file"] = fname

        fname_name, fname_ext = os.path.splitext(fname)
        dict_return["file_name"] = fname_name
        dict_return["file_ext"] = fname_ext

        return dict_return

    def get_faces(self, imgFile):
        pp = self.parsePath(imgFile)
        faceFilesPattern = "".join([pp["last_dir"],"_#_",pp["file_name"],"_*"])
        faceFilesDir = os.path.join(self.getConfig('path','faces_dir'),pp["last_dir"])
        if os.path.exists(faceFilesDir):
            faceFilesPath = os.path.join(faceFilesDir,faceFilesPattern)
            faceFiles = glob.glob(faceFilesPath)
            if len(faceFiles) > 0:
                return faceFiles
        return []
    
    def save_faces(self, imgFile, overwrite = False): 
        print("func: save_faces() is processing file:", imgFile) 
        if os.path.isfile(imgFile) is False:
            logging.error("file is not existing: "+ imgFile)
            return None

        if len(self.get_faces(imgFile)) > 0 and overwrite == False:
            logging.info("faces are existing, no need to overwrite, will skip it: "+ imgFile)
            return self.get_faces(imgFile)

        pp = self.parsePath(imgFile)
        faceFilesPattern = "".join([pp["last_dir"],"_#_",pp["file_name"],"_*"])
        faceFilesDir = os.path.join(self.getConfig('path','faces_dir'),pp["last_dir"])
        if os.path.exists(faceFilesDir) is False:
            logging.warn("faces' savePath is not existing, need to create. "+ faceFilesDir)
            os.mkdir(faceFilesDir)
    
        bgrImg = cv2.imread(imgFile)
    
        if bgrImg is None:
            logging.error("source image is invalid, cv2 cannot read the file."+ imgFile)
            return None

        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        facesrectangles = self.face_detector(rgbImg,1)
        if len(facesrectangles) <= 0:
            logging.error("cannot detect the face(s) : " + imgFile)
            return None
        
        faces = []
        for k,d in enumerate(facesrectangles):
            shape = self.landmark_predictor(rgbImg, d)
            print(k,":", d)
            tlx = d.left() 
            tly = d.top() 
            brx = d.right() 
            bry = d.bottom()

            faceImg = cv2.imread(imgFile)
            face = faceImg[tly:bry, tlx:brx]
            face_resized = cv2.resize(face,(self.getConfig('face','face_width'),self.getConfig('face','face_height')),interpolation=cv2.INTER_AREA)
            if face_resized is None:
                logging.error("face_resized crop failed")
                return None
            dstFileName = "".join([pp["last_dir"],"_#_",pp["file_name"],"_",str(k),pp["file_ext"]])
            dstFileSavePath = os.path.join(faceFilesDir,dstFileName)
            cv2.imwrite(dstFileSavePath, face_resized,[int(cv2.IMWRITE_JPEG_QUALITY), self.getConfig('face','face_jpeg_quality')])
            faces.append(dstFileSavePath)
        
        return faces

    
    def scanFaceSourceDir(self):
        srcDir = self.getConfig("path", "source_dir")
        if os.path.isdir(srcDir) is False:
            logging.error("source dir is not existing:" + srcDir)
            return []

        imagesNeed = []
        for rt, dirs, files in os.walk(srcDir):
            if rt is None:
                return None
            
            for d in dirs:
                fpath = os.path.join(srcDir,d,"*.jpg")
                files = glob.glob(fpath)
                for f in files:
                    f_faces = self.get_faces(f)
                    if len(f_faces) <= 0:
                        imagesNeed.append(f) 
        return imagesNeed    
    
    def scanFaceRecognitionDir(self):
        srcDir = self.getConfig("path", "recognition_dir")
        if os.path.isdir(srcDir) is False:
            logging.error("recognition dir is not existing:" + srcDir)
            return None
        
        srcDir = os.path.abspath(srcDir)
        files = glob.glob(srcDir + "/*.jpg")
        if len(files) > 0:
            for f in files:
                fpath = os.path.join(srcDir,f)
                f_faces = self.get_faces(fpath)
                print("iii",f_faces)
                if len(f_faces) <= 0:
                    self.save_faces(fpath)
        return None    

    def get_face_descriptor(self,imgOneFaceFilePath):
        if imgOneFaceFilePath is None:
            logging.error("image path cannot be empty.")
            return False

        if os.path.isfile(imgOneFaceFilePath) is False:
            logging.error("image path is not existing."+imgOneFaceFilePath)
            return False

        sha1 = hashlib.sha1()  
        sha1.update(imgOneFaceFilePath.encode("utf8"))
        sha1_imgPath = sha1.hexdigest() 

        face_d = self.get_np_cache(sha1_imgPath)
        if face_d is not None:
            return face_d

        bgrImg = cv2.imread(imgOneFaceFilePath) 
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
        facerect = self.face_detector(rgbImg,1)
        if len(facerect) <= 0:
            print("cannot detect the face(s)",imgOneFaceFilePath)
            logging.warn("cannot detect the face(s)" + imgOneFaceFilePath)
            return False
        print(facerect)
        for k,d in enumerate(facerect):
            print('one face rectangle:',d)
            shape2 = self.landmark_predictor(rgbImg, d)
            face_descriptor = self.face_recognition_model.compute_face_descriptor(rgbImg, shape2, self.getConfig('face','face_compute_descriptor_call'))

        self.set_np_cache(sha1_imgPath,face_descriptor)
        return face_descriptor     

    
    def get_face_recognition(self, imgPath):
        imgPath_sha1 = None
        with open(imgPath,'rb') as f:
            sha1 = hashlib.sha1()
            sha1.update(f.read())
            imgPath_sha1 = sha1.hexdigest()
        
        cfaces = self.get_np_cache(imgPath_sha1)
        if cfaces is not None:
            for cface in cfaces:
                if cface["face_candidate_distance"] > 0.6:
                    print("\nThe person : \033[1;47;41m ","CANNOT BE RECOGNIZED.", "\033[0m",str(cface["face_candidate_distance"]), " <== ",cface["face_candidate_image"]," <== ",imgPath)
                    print("But, this person is most likely to be: \033[1;30;43m ",cface["face_recognized_name"], "\033[0m",str(cface["face_candidate_distance"])," <== ",cface["face_candidate_image"], " <== ",imgPath)
                else:
                    print("\nThe person is: \033[1;47;46m ",cface["face_recognized_name"], "\033[0m",str(cface["face_candidate_distance"])," <== ", cface["face_candidate_image"]," <== ",imgPath)
                
            return {'sha1_hash':imgPath_sha1, 'faces': cfaces}

        if self.candidates_face_descriptors is None:
            print("candidates_face_descriptors is empty, will run")
            candidates_face_descriptors = get_candidates_desciptors(self.candidates_list)

        Faces_ImgPath = self.get_faces(imgPath)
        if len(Faces_ImgPath) <= 0:
            Faces_ImgPath = self.save_faces(imgPath)
            if len(Faces_ImgPath) <= 0:
                print("cannot detect the face(s) on the image: " , imgPath)
                logging.WARN("cannot detect the face(s) on the image: " + imgPath)
                return None       

        faces_result = []
        for faceIP in Faces_ImgPath:
            test_descriptor = self.get_face_descriptor(faceIP)
            v_test_descriptor = np.array(test_descriptor)
            dist = []
            for i in self.candidates_face_descriptors:
                dist_test = np.linalg.norm(i - v_test_descriptor)
                dist.append(dist_test)

            c_d = dict(zip(self.candidates_list,dist))

            cd_sorted = sorted(c_d.items(), key=lambda d:d[1])
            candImg = cd_sorted[0][0]
            person_name = candImg[0:candImg.find('_#_')]
            frec = {}
            frec["face_recognized_name"] = person_name
            frec["face_candidate_image"] = candImg
            frec["face_candidate_distance"] = cd_sorted[0][1]
            if cd_sorted[0][1] > 0.6:
                frec["face_recognition_success"] = 0
                print("\nThe person : \033[1;47;41m ","CANNOT BE RECOGNIZED.", "\033[0m",str(cd_sorted[0][1]), " <== ",candImg," <== ",imgPath)
                print("But, this person is most likely to be: \033[1;30;43m ",person_name, "\033[0m",str(cd_sorted[0][1])," <== ",candImg, " <== ",imgPath)
            else:
                frec["face_recognition_success"] = 1
                print("\nThe person is: \033[1;47;46m ",person_name, "\033[0m",str(cd_sorted[0][1])," <== ", candImg," <== ",imgPath)
            faces_result.append(frec)
        self.set_np_cache(imgPath_sha1,faces_result)
        return {'sha1_hash':imgPath_sha1, 'faces': faces_result}










