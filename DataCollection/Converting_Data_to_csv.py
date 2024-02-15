
import cv2
import os
import mediapipe as mp
import pandas as pd
import pickle
hands=mp.solutions.hands.Hands(max_num_hands=2,min_detection_confidence=0.3)

# dir="./DataCollection/"
count=0;

new_data_frame=pd.DataFrame([])

for i in range(0,9):
    dir=f"./DataCollection/{i}"
    
    for j in  os.listdir(dir):
        # print(dir)
        image_path=os.path.join(dir,j)
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        # print(image)
        
        handlandmarks=hands.process(image)
        print(image_path)
        if handlandmarks.multi_hand_landmarks:
            landmarks_data=handlandmarks.multi_hand_landmarks
            data=[]
            for k in landmarks_data:
                for landmark in k.landmark:
                    data.append(landmark.x)
                    data.append(landmark.y)
                    
                data.append(i)       
                dataframe=pd.DataFrame(data).transpose()
                
            new_data_frame=pd.concat([new_data_frame,dataframe])
         
            
print(new_data_frame)

pickle_file_path = "data.pkl"

with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(new_data_frame, pickle_file)

        
        

        
        # print(image_path)
          
        # print(landmarks_data)
        # break
        
     
    