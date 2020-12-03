## Hi 👋

来都来了，不点个小星星吗？

Welcome to star this repo

凌空画笔 [<a href="https://www.bilibili.com/video/BV15V411a7WB">Demo</a>]

README [<a href="README.md">EN</a>|<a href="README_CN.md">CN</a>]

## Description

凌空手势识别和绘制，默认手势1是画笔，手势2是更换颜色，手势5是清空画板
显示基于OpenCV


## Change Log

### v3.0

该版本项目基于<a href="https://github.com/insigh1/GA_Data_Science_Capstone/">GA_Data_Science_Capstone</a>

用Yolo_v5识别手势和食指进行绘制，请自行手势数据集并进行标注，数据预处理在01和02文件中
该项目可移植到树莓派上运行，利用树莓派收集图像，推流到电脑进行推理，有延迟

#### How to run

```sh
cd v3.0
pip install -r requirements.txt
jupyter notebook

# open and run 01_image_processing_and_data_augmentation.ipynb

# run labelImg to label data 1, 2, 5, forefinger

python 02_munge_data.py

# train model
python train.py --img 512 --batch 16 --epochs 100 --data config.yaml --cfg models/yolov5s.yaml --name yolo_example
tensorboard --logdir runs/

# run use pc cam
python detect.py --weights weights/best.pt --img 512 --conf 0.3 --source 0

# run use raspi
# run on raspi
sudo raspivid -o - -rot 180 -t 0 -w 640 -h 360 -fps 30|cvlc -vvv stream:///dev/stdin --sout '#standard{access=http,mux=ts,dst=:8080}' :demux=h264  
# run on pc
python detect.py --weights runs/exp12_yolo_example/weights/best.pt --img 512 --conf 0.15 --source http://192.168.43.46:8080/
```

### v2.0

基于OpenCV和凸包检测的手势识别
肤色检测+凸包+数轮廓线个数（统计手指数量）

#### How to run

```sh
cd v2.0
python gesture.py
```



### v1.0

基于OpenCV的肤色检测+凸包


#### How to run
```sh
cd v1.0
python main.py
```



