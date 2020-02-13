import os

path1 = 'C:/Users/Salemd/Downloads/'
path2 = 'VISDrone/Task 1 Object Detection in Images/VisDrone2019-DET-train/images/'
path3 = 'VISDrone/Task 1 Object Detection in Images/VisDrone2019-DET-val/images/'

txtfile = open(path1 + 'train.txt', 'a')
for filename in os.listdir(path1 + path2):
    txtfile.write(path1 + path2 + filename + '\n')

txtfile = open(path1 + 'val.txt', 'a')
for filename in os.listdir(path1 + path3):
    txtfile.write(path1 + path3 + filename + '\n')

