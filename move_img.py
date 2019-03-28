import os

folders = ['angry_faces','conv','crying_faces','happy_faces','happy_faces_1','sad_faces']
path = r'C:\Users\AArup\Desktop\ALL\YOLO-V2\images'

i = 0
for folder in folders:
    for images in os.scandir(folder):
        os.rename(images.path,os.path.join(path,'{:06}.jpg'.format(i)))
        i += 1 
