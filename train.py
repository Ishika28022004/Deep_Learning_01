import cv2
import numpy as np
import os

faces = []
IDs = []
def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    
    for imagePath in imagePaths:
        faceImg = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)  # Use grayscale
        if faceImg is None:
            print(f"Could not read image: {imagePath}")
            continue
        faceNp = np.array(faceImg, 'uint8')
        
        try:
            ID = int(os.path.split(imagePath)[-1].split('.')[0].split('_')[1])
        except Exception as e:
            print(f"Filename format issue in: {imagePath} — {e}")
            continue
        
        faces.append(faceNp)
        IDs.append(ID)
    
    return np.array(IDs, dtype=np.int32), faces

# Load data
IDs, faces = getImagesWithID('images')

# Create recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train with properly formatted labels
recognizer.train(faces, IDs)

# Save the trained model
recognizer.save('trainingData.yml')
