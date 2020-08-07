import cv2
import numpy as np
face_classifier=cv2.CascadeClassifier('C:/Users/Lenovo/AppData/Local/Programs/Python/Python38/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def face_extractor(image):
    grey=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grey,1.3,5)
    if faces is None:
        return None
    for(x,y,w,h) in faces:
        cropped=image[y:y+h,x:x+w]
        return cropped


cap=cv2.VideoCapture(0)
count=0
while True:
    ret,frame=cap.read()
    if face_extractor(frame) is not None:
        count=count+1
        face=cv2.resize(face_extractor(frame),(200,200))
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        file_name_path='E:/ ML project/face1/'+str(count)+'.jpg'
        cv2.imwrite(file_name_path,face)
        cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("face cropper",face)
    else:
        print("face not found")
        pass
    if cv2.waitKey(1)==13 or count==100:
        break    
cap.release()
cv2.destroyAllWindows()