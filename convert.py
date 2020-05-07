import os
import cv2
from tqdm import tqdm

srcpath = "train"
destpath = "data"

folders = os.listdir(srcpath)
#print(files)

for fol in tqdm(folders):
    files = os.listdir(srcpath + '\\' + fol)
    folder_path = srcpath + '\\' + fol
    save_path = destpath + '\\' + fol
    if(not os.path.isdir(save_path)):
        os.mkdir(save_path)
    for f in files:
        img_loc = folder_path + '\\' + f
        img = cv2.imread(img_loc,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200,200))
        img = cv2.bitwise_not(img)
        cv2.imwrite(save_path + '\\' + f, img)