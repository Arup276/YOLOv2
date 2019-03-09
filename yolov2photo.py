# importing the dependencies
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import argparse

# the cfg file and weights location change this thing according to your model location
model = {"model": "cfg/yolov2.cfg",
           "load": "yolov2.weights",
           "threshold": 0.4,
         "gpu":0}

# creating the object
tfnet = TFNet(model)

# to get the image path
parser = argparse.ArgumentParser()
parser.add_argument('--img',type=str,help='path of the image')
arg = parser.parse_args()


imgc = cv2.imread(arg.img) # read the image
imgcv = cv2.cvtColor(imgc,cv2.COLOR_RGB2BGR)
result = tfnet.return_predict(imgcv) # predict the classes and cordinates of the oject

# This is for draw the bounding box around the predicted classes 
tl = []
br = []
labels = []
confidence = []
cf_cor = []

for i in range(len(result)):
    topleft = (result[i]['topleft']['x'],result[i]['topleft']['y']) # to get the topleft cordinates in a tuple
    bottomright = (result[i]['bottomright']['x'],result[i]['bottomright']['y']) # to get the bottomright cordinates in a tuple
    label = (result[i]['label']) # to get the labels from the predicted class ,it's in the form of dictionary
    conf = str(round(result[i]['confidence'],2))
    
    # to add the confidence score in middle
    st = result[i]['topleft']['x'] 
    nd = result[i]['bottomright']['x']
    mid_x = (nd-st)//2 + st # mid point of the top box line
    mid_y = result[i]['topleft']['y']
    
    cordi = (mid_x,mid_y)
    cf_cor.append(cordi)                 
    tl.append(topleft) # append the tuples in the list
    br.append(bottomright)
    labels.append(label)
    confidence.append(conf)
    img2 = cv2.rectangle(imgcv,tl[i],br[i],(0,255,0),3) # draw rectangles around the classes here we pass image,topleft cordinates ,bottomright cordinates ,which colour box we want and how thik the line
    img2 = cv2.putText(imgcv,labels[i],tl[i],cv2.FONT_HERSHEY_COMPLEX,1, (255 ,0 ,0), 2) # putting the label on the topleft corner
    img2 = cv2.putText(imgcv,confidence[i],cf_cor[i],cv2.FONT_ITALIC,1, (255 ,0 ,0), 2) # putting the confidence score
    

    
img3 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)  # convert the image in RGB format
cv2.imshow('prediction',img3)
cv2.waitKey(0) # waitkey for hold the image in display until user press any key
