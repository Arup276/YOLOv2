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
Here the most interesting part comes...\

If you want your model to render a video then copy the video and paste it under the darkflow-master folder.
and use this command
```
python flow --model cfg/yolo.cfg --load bin/yolov2.weights --demo videofile.mp4 --gpu 1.0 --saveVideo
```
#### if you get some error like `ModuleNotFoundError: No module named 'nms'` then download the repo from [here](https://github.com/thtrieu/darkflow) and run build_ext properly.

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

just use ```camera``` keyword.


## If you want to detect object from a image then go for it,
Open you jupyter notebook and start typing...
```
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolov2.weights", "threshold": 0.3}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```

### Their is no code for run yolov2 on image from cmd but here is the solution just run `python photoshow.py` ,before the open the file and modify the `option` according to your cfg and weight location.


```# importing the dependencies
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import argparse

# the cfg file and weights location change this thing according to your model location
model = {"model": "cfg/yolov2.cfg",
           "load": "yolov2.weights",
           "threshold": 0.4}

# creating the object
tfnet = TFNet(model)

# to get the image path
parser = argparse.ArgumentParser()
parser.add_argument('--img',type=str,help='path of the image')
arg = parser.parse_args()

imgcv = cv2.imread(arg.img) # read the image
result = tfnet.return_predict(imgcv) # predict the classes and cordinates of the oject

# This is for draw the bounding box around the predicted classes 
tl = []
br = []
labels = []
for i in range(len(result)):
    topleft = (result[i]['topleft']['x'],result[i]['topleft']['y']) # to get the labels from the predicted class ,it's in the form of dictionary
    bottomright = (result[i]['bottomright']['x'],result[i]['bottomright']['y'])
    label = (result[i]['label'])
    tl.append(topleft)
    br.append(bottomright)
    labels.append(label)
    img2 = cv2.rectangle(imgcv,tl[i],br[i],(0,255,255),5) # draw rectangles around the classes here we pass image,topleft cordinates ,bottomright cordinates ,which colour box we want and how thik the line
    img2 = cv2.putText(imgcv,labels[i],tl[i],cv2.FONT_HERSHEY_COMPLEX,1, (0 ,0 ,0), 2) # putting the label on the topleft corner
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)  # convert the image in RGB format
cv2.imshow('prediction',img2)
cv2.waitKey(0) # waitkey for hold the image in display until user press any key

```

It will predict the object with confident score in that given image.   
Now enjoy ✌.


