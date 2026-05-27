#load
import cv2
#load the smile haar cascade model
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")

#input the image
image=cv2.imread("model_1.jpg")

#convert the colour
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#detect the face and the smile
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.8,minNeighbors=5)

#rectangle box
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),3)

    #face region
    roi_gray=gray[y:y+h,x:x+w]
    roi_color=image[y:y+h,x:x+w]
    #detect the smile

    smiles=smile_cascade.detectMultiScale(roi_gray,scaleFactor=1.8,minNeighbors=20)

    for(sx,sy,sw,sh) in smiles:
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,255,0),3)

        cv2.imshow("face and smile detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
