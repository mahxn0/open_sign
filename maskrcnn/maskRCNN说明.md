#文件结构
训练数据集：历史训练数据均保存在data文件夹下，当前训练数据集可放在maskRCNN目录下
训练模型：训练模型均保存在logs文件夹下

参考
http://blog.csdn.net/l297969586/article/details/79140840

#训练及测试过程
##1. 数据准备
1. 训练数据集目录
训练集文件 ----|------json 存放.json标签文件
               |------mask
               |------rgb  存放.jpg图片
.jpg图片数据统一命名为rgb_0,jpg, rgb_1.jpg...格式
对应的.json标签文件统一命名为rgb_0,json, rgb_1.json...格式
注意图像尺寸应统一尺寸（本说明为256x256）

2. 生成标签文件
json2label.sh
根据具体训练集数据修改训练集路径及i值
运行：
```
bash json2label.sh
```
运行后将会在数据集文件夹下的json文件夹中生成标签，一个json文件对应一个标签文件夹

3. 16位mask灰度图转8位灰度图
同样的，修改mask_label.cpp中的路径及i值 
运行：
```
g++ -o mask_label mask_label.cpp `pkg-config opencv --cflags --libs`
```
执行生成的可执行文件后，会将标签文件中的mask灰度图转为8位保存到训练集文件夹下的mask文件夹中

##2. 训练模型 
train_shapes.py
（1）
GPU_COUNT = 1 #GPU个数
IMAGES_PER_GPU = 8 #每个GPU处理的图片数目，当out of memory时可适当调小这个参数
（2）按实际类别数修改
NUM_CLASSES = 1 + 3  # background +  classes 即1+类别数
（3)修改为实际照片尺寸，本例中为256x256
IMAGE_MIN_DIM = 256
IMAGE_MAX_DIM = 256
（4）修改load_shapes函数中的标签名称，标签编号从1开始
e.g.
self.add_class("shapes", 1, "needle")
self.add_class("shapes", 2, "pin")
（5）修改load_mask函数中的标签
（6）修改训练集路径dataset_root_path

python3 train_shapes.py
训练生成的模型保存在logs文件夹下，默认名称为shapes+训练时间

##3. 测试模型
test_shapes.py
参考train_shapes.py修改标签名及路径
修改需要测试的模型路径model_path
修改测试图片路径root_dir及输出结果保存路径output_dir

python3 train_shapes.py

