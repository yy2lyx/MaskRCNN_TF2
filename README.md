# MaskRCNN_TF2

MaskRCNN 在TF2.x情况下跑通自己的数据集
* mrcnn 替换掉官方的mrcnn文件夹即可开箱使用
* labelme2coco 将labelme打标工具生成的json变成coco形式的json文件


训练：`python train.py train --dataset=./dataset --weights=coco` （注意这里要把数据集放在dataset目录下哦）
预测：`python pred.py`