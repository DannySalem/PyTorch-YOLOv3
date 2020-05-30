import os
import cv2


path2 = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/images/'
path1 = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/annotations/'

txtpath = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/labels/'


# 684,8,273,116,0,0,0,0
# 684/w, 8/h, 273/w, 116/h

dataset_folder = 'val/'

for filename in os.listdir(path1 + dataset_folder):
    OGtxtfile = open(path1 + dataset_folder + filename, 'r')
    NEWtextfile = open(txtpath + dataset_folder + filename, 'a')
    im = cv2.imread(path2 + dataset_folder + filename[:-4] + '.jpg')

    while True:
        line = OGtxtfile.readline()
        if not line:
            break
        y = line.split(',')
        category = y[5]
        y = y[0:4]
        height, width, _ = im.shape
        boxwidth = int(y[2])
        boxheight = int(y[3])   

        y[0] = (int(y[0]) + boxwidth/2) / width
        y[1] = (int(y[1]) + boxheight/2) / height
        y[2] = int(y[2]) / width
        y[3] = int(y[3]) / height

        output = [category] + [f"{x:.6f}" for x in y]
        output = ' '.join(output) + ' \n'

        NEWtextfile.write(output)
        #print(output)
    NEWtextfile.close()
