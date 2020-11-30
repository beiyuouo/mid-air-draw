# mid-air-draw

mid-air-draw[<a href="https://www.bilibili.com/video/BV15V411a7WB">Demo</a>]

Welcome to star this repo

## v1.0
肤色检测+凸包

```sh
cd v1.0
python main.py
```

## v2.0
肤色检测+凸包+数轮廓线个数（统计手指数量）

### How to run

```sh
cd v2.0
python gesture.py
```


## v3.0

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