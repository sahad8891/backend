from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
import joblib
from face_features import extract_features

app = FastAPI()

model = joblib.load("models/twsm_model.pkl")

@app.post("/analyze")

async def analyze(video: UploadFile):

    file_bytes = await video.read()

    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    features = extract_features(img)

    prediction = model.predict([features])

    return {"risk": prediction[0]}


@app.post("/questionnaire")

async def questionnaire(data: dict):

    answers = np.array(data["answers"])

    prediction = model.predict([answers])

    return {"risk": prediction[0]}
