import cv2
import mediapipe as mp
import numpy as np

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

def extract_features(image):

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return np.zeros(10)

    landmarks = results.multi_face_landmarks[0].landmark

    features=[]

    for i in range(10):
        features.append(landmarks[i].x)
        features.append(landmarks[i].y)

    return np.array(features)
