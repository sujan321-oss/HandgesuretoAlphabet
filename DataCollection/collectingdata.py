import cv2
import numpy as np 
import os
import time

cap=cv2.VideoCapture(0)

for i in range (0,9):
    dir=f'./DataCollection/{i}'
    if not os.path.exists(dir):
        os.makedirs(dir)
    print("Collecting for ",i)
        
    while 1:
        ret,frame=cap.read()
        cv2.imshow("frame",frame)
        cv2.putText(frame,"ready ? press q",(25,25),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
        if cv2.waitKey(1) &  0xFF == ord('q'): #press 'q' to capture the image
            break
    
    for j in range (0,50):
        file_path=os.path.join(dir,f"./{j}.jpg")
        ret,frame=cap.read()
        cv2.imshow("frame",frame)
        cv2.imwrite(file_path,frame)
        cv2.waitKey(20)
  

#  classes=['stop','fuck','rock_it','DekhLunga_madarchod','namaste','feast','i_love_you','susu_jana_hey_mereko','itney_mey_jayega_tera']
    
  
        
        
    