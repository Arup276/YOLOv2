# importing the dependencies
import cv2
import matplotlib.pyplot as plt
from darkflow.net.build import TFNet
import cv2

# the cfg file and weights location
options = {"model": "cfg/yolov2.cfg",
           "load": "yolov2.weights",
           "threshold": 0.4}

# creating the object
tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_horses.jpg") # read the image
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
