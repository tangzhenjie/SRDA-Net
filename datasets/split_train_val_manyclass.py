import os
import shutil

def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)
        if not os.path.exists(fpath):
            os.makedirs(fpath)
        shutil.copyfile(srcfile,dstfile)
        print("copy %s -> %s"%(srcfile,dstfile))

label_dir = "./labels/"
image_dir = "./images/"

label_paths = os.listdir(label_dir)
lenth = len(label_paths)
tag = lenth // 2
print(lenth)
print(tag)
for index, path_item in enumerate(label_paths):
	result_image = "./train/images/"
	result_label = "./train/labels/"
	if index >= tag:
		result_image = "./val/images/"
		result_label = "./val/labels/" 
	mycopyfile(label_dir + path_item, result_label + path_item)
	image_name = path_item[:-9] + "IRRG.tif"
	mycopyfile(image_dir + image_name, result_image + image_name)

