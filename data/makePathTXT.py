import os

path1 = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/images/train/'
path2 = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/images/val/'

output1 = '/data/salemd/PyTorch-YOLOv3/data/visdrone/images/train/'
output2 = '/data/salemd/PyTorch-YOLOv3/data/visdrone/images/val/'

txtpath = 'C:/Projects/PyTorch-YOLOv3/data/visdrone/images/'

txtfile = open(txtpath + 'train.txt', 'a')
for filename in os.listdir(path1):
    txtfile.write(path1 + filename + '\n')

txtfile = open(txtpath + 'val.txt', 'a')
for filename in os.listdir(path2):
    txtfile.write(path2 + filename + '\n')


txtfile = open(txtpath + 'train_Linux.txt', 'a')
for filename in os.listdir(path1):
    txtfile.write(output1 + filename + '\n')

txtfile = open(txtpath + 'val_Linux.txt', 'a')
for filename in os.listdir(path2):
    print(output2)
    txtfile.write(output2 + filename + '\n')
