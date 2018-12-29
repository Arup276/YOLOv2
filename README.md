## As you know that YOLO(You Only Look Once) is real time object detection algorithm.

## You can implement this by own just follow this.

### Step1: Dependencies.
1.python 3.6 ([download](https://www.python.org/downloads/)chose 3.6.0).  
2.OpenCV([downloas](https://www.lfd.uci.edu/~gohlke/pythonlibs/)).  
3.tensorflow (If you have gpu then use gpu [version](https://www.tensorflow.org/install/))  

### Step2: Clone and Download this repository,Or you can download it from [here](https://github.com/thtrieu/darkflow).

### Step3: Build the cython extensions.
Open cmd at darkflow-master folder and type
```
python setup.py build_ext --inplace
```

### Step4: Download the weights.
1. Downloads the weight according to your model here we will use YOLOv2 so [weights](https://pjreddie.com/darknet/yolo/)(Download YOLOv2 608x608 weight).  

2. Create a folder in darkflow-master folder and name it bin.  

3. Paste the downloaded weight in this folder(or you can direct download in this folder😁).  

### Step5: Nothing just start it.
Here the most interesting part comes traningggg...\

If you want your model to render a video then copy the video and paste it under the darkflow-master folder.
and use this command
```
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --gpu 1.0 --saveVideo
```

#### Code Description.
python flow --model cfg/yolo.cfg :- Which model you want to use give it path to the model.  

--load bin/yolov2.weight :- Path of the weight check the downloaded weight name if this is yolo.weight then make yolov2.weight to yolo.weight is the above code.

--demo videofile.mp4  :- video file name name.

--gpu 1.0 :- tensorflow gpu if you have tensorflow gpu then give it leave it.

--saveVideo :- This is for save the video file, your rended video file will be video.avi named.

### Now if you want your web camera to detect object then run this command
```
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo camera --gpu 1.0 --saveVideo
```

just use ```camera``` keyword./

Now enjoy ✌.


