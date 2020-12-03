## Hi 👋

Come here, don’t you star this progect? & Forgive my pool English.

Welcome to star this repo!

Mid-air brush [<a href="https://www.bilibili.com/video/BV15V411a7WB">Demo</a>]

README [<a href="README.md">EN</a>|<a href="README_CN.md">CN</a>]

## Description

Mid-air gesture recognition and drawing, the default gesture 1 is a brush, gesture 2 is to change the color, and gesture 5 is to clear the drawing board
Display based on OpenCV.


## Change Log

### v3.0

This version of the project is based on <a href="https://github.com/insigh1/GA_Data_Science_Capstone/">GA_Data_Science_Capstone</a>

Use Yolo_v5 to recognize gestures and index fingers for drawing. Please make your own gesture dataset and label them. Data preprocessing is in files 01 and 02.
The project can be run on Raspberry Pi, use the Raspberry Pi to collect images and push them to the computer for reasoning, there is a delay.

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

Gesture recognition based on OpenCV and convex hull detection.
Skin color detection + convex hull + number of contour lines (count the number of fingers).

#### How to run

```sh
cd v2.0
python gesture.py
```



### v1.0

Skin color detection + convex hull based on OpenCV.


#### How to run
```sh
cd v1.0
python main.py
```



