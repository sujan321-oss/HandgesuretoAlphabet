
# Sign Language to Words Process

## 1. Data Collection
The first step involves collecting data using the Mediapipe library, which is designed for real-time hand tracking. This library captures hand gestures and extracts key data points, such as the positions of fingers and hand movements.
The collected data is recorded and stored in a CSV file. This dataset serves as the foundation for training the machine learning model.
## 2. Machine Learning Model Training
The dataset is used to train a machine learning algorithm, specifically the Decision Tree Classifier. This classification algorithm is chosen for its ability to make decisions based on the input features derived from the hand gestures recorded by Mediapipe.
During the training process, the model learns to associate specific hand gestures with corresponding words or letters. This training phase is crucial for the model to generalize and recognize new, unseen gestures accurately.
## 3. Implementation with OpenCV
Once the model is trained, it is implemented using the OpenCV (Open Source Computer Vision) library. OpenCV provides tools for image processing and computer vision applications.
The trained machine learning model is integrated into an OpenCV-based system, allowing it to interpret real-time hand gestures captured by a camera or another input device.
As users perform sign language gestures in front of the camera, the system processes the hand movements and translates them into corresponding words or letters based on the learned associations from the training phase.


# Cone this project
```bash
git clone https://github.com/sujan321-oss/HandgesuretoAlphabet.git

```

```bash 
cd handgesturetoalphabet/prediction

```

```bash 

python prediction.py

```




