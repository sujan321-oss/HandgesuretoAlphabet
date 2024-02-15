
import numpy as np
import mediapipe as mp 
import pickle
import cv2
cap=cv2.VideoCapture(0)

hands=mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.3)

classes=['stop','fuck','rock_it','DekhLunga_madarchod','namaste','feast','i_love_you','susu_jana_hey_mereko','itney_mey_jayega_tera']


Prediction_array=[]

# loading the model
model_path="/Users/khumapokharel/Desktop/deepLearning/HandgesuretoAlphabet/ModelTraining/model.pkl"

with open(model_path,"rb") as file:
    model=pickle.load(file)
    



while True:
    Prediction_array=[]
    ret,frame=cap.read()
    h,w,c=frame.shape
    

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    detected_hands=hands.process(gray_frame)
    
    
    hand=frame
    if detected_hands and detected_hands.multi_hand_landmarks:
        Prediction_array=[]
        for i in detected_hands.multi_hand_landmarks:
            prev = (int(i.landmark[0].x * w), int(i.landmark[0].y * h))
            Prediction_array.append(i.landmark[0].x)
            Prediction_array.append(i.landmark[0].y)
            
            for landmark in i.landmark[1:]:
                Prediction_array.append(landmark.x)
                Prediction_array.append(landmark.y)
                
                hand=cv2.line(hand,prev,(int(landmark.x*w),int(landmark.y*h)),(255,255,255),1)
                hand=cv2.circle(hand,(int(landmark.x*w),int(landmark.y*h)),1,(255,0,0),2)
                prev=(int(landmark.x*w),int(landmark.y*h))
    
    
    
        last_hand_mark=(int(detected_hands.multi_hand_landmarks[0].landmark[-1].x*w),int(detected_hands.multi_hand_landmarks[0].landmark[-1].y*h))
        first_hand_mark=(int(detected_hands.multi_hand_landmarks[0].landmark[0].x*w),int(detected_hands.multi_hand_landmarks[0].landmark[0].y*h))
        hand=cv2.line(hand,first_hand_mark,last_hand_mark,(255,255,255),1)

    
    
     
        Prediction_array=np.array([Prediction_array])
        prediction=model.predict(Prediction_array)
       
        final_prediction=classes[prediction[0]]
        cv2.putText(hand,final_prediction,(100,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
        

    
    cv2.imshow("frame",hand)
    
    
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    

    
    

    
cap.release()
cv2.destroyAllWindows()

