import cv2
import numpy as np 

#get image classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#read images
sombrero = cv2.imread('mariachisombrero.png')

#get shape of sombrero
original_sombrero_h,original_sombrero_w,sombrero_channels = sombrero.shape

#convert to gray
sombrero_gray = cv2.cvtColor(sombrero, cv2.COLOR_BGR2GRAY)

#create mask and inverse mask of sombrero
ret, original_mask = cv2.threshold(sombrero_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

#read video
cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]


while 1:
    ret, img = cap.read()

    #convert ro gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        #coordinates of face
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        #hat relation to face by scaling
        sombrero_width = int(1.5 * face_w)
        sombrero_height = int(sombrero_width * original_sombrero_h / original_sombrero_w)

        #setting location of coordinates of sombrero
        sombrero_x1 = face_x2 - int(face_w/2) - int(sombrero_width/2)
        sombrero_x2 = sombrero_x1 + sombrero_width
        sombrero_y1 = face_y1 - int(face_h*1.25)
        sombrero_y2 = sombrero_y1 + sombrero_height 

        #check to see if out of frame
        if sombrero_x1 < 0:
            sombrero_x1 = 0
        if sombrero_y1 < 0:
            sombrero_y1 = 0
        if sombrero_x2 > img_w:
            sombrero_x2 = img_w
        if sombrero_y2 > img_h:
            sombrero_y2 = img_h

        #if out of frame changes
        sombrero_width = sombrero_x2 - sombrero_x1
        sombrero_height = sombrero_y2 - sombrero_y1

        # dsize
        dsize = (sombrero_width, sombrero.shape[0])   

        #resize sombrero to fit on face

        #sombrero = cv2.resize(sombrero [sombrero_width,sombrero_height], interpolation = cv2.INTER_AREA)
        #mask = cv2.resize(original_mask, (sombrero_width,sombrero_height))
        #mask_inv = cv2.resize(original_mask_inv, (sombrero_width,sombrero_height))

        sombrero= cv2.resize(sombrero, dsize, interpolation = cv2.INTER_AREA)
        mask= cv2.resize(original_mask, dsize, interpolation = cv2.INTER_AREA)
        mask_inv= cv2.resize(original_mask_inv, dsize, interpolation = cv2.INTER_AREA)

        mask_inv= cv2.resize(original_mask_inv, dsize, interpolation = cv2.INTER_AREA)


        #take ROI for sombrero from background that is equal to size of sombrero image
        roi = img[sombrero_y1:sombrero_y2, sombrero_x1:sombrero_x2]
        #original image in background (bg) where sombrero is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        #roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(sombrero,sombrero,mask =mask)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[sombrero_y1:sombrero_y2, sombrero_x1:sombrero_x2] = dst
        break
        
    #display image
    cv2.imshow('img',img) 

    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): 
        break

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windows

        