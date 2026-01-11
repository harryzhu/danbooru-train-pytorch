import os
from PIL import Image
import cv2

def rm_broken_image(dpath):
	for root, dirs, files in os.walk(dpath, True):
		for f in files:
			if f[-4:] != ".png":
				continue
			fpath = os.path.join(root,f)

			try:
				img = cv2.imread(fpath)
				if img is None:
					print(f'ERROR:{fpath}')
			except Exception as err:
				print(fpath)
				print(err)


rm_broken_image("d:/app/t3/data/images")