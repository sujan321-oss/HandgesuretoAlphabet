import cv2
import mediapipe as mp 
import pandas as pd
import os
import pickle

hands=mp.solutions.hands.Hands()


# handdata=[]



dataframe=pd.DataFrame([])

for file in range(0,9):
    dir=f"./DataCollection/{file}"
    
    
    
    for j in os.listdir(dir):
        image_path=os.path.join(dir,j)
        print(image_path)
        image=cv2.imread(image_path)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        data=hands.process(image)
        
        if data.multi_hand_landmarks:
            for i in data.multi_hand_landmarks:
                handdata=[]
                for k in i.landmark:
                    handdata.append(k.x)
                    handdata.append(k.y)
            handdata.append(file)
            
            handdata_frame=pd.DataFrame(handdata).transpose()
        
            dataframe=pd.concat([dataframe,handdata_frame],axis=0)
            # print(dataframe)
    # break
            
        
print(dataframe)

pickle_file_path = "final_data.pkl"

with open(pickle_file_path, 'wb') as pickle_file:
    pickle.dump(dataframe, pickle_file)

        
        
        
        





    
    
