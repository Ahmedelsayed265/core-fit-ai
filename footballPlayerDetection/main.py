from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import base64
import requests
import shutil

app = FastAPI()
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mount static files for image access
app.mount("/static", StaticFiles(directory="static"), name="static")

load_dotenv()
# Groq API Key
GROQ_API_KEY = os.getenv("key")  
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def detect_faces(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        return None, img_rgb
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        return None, img_rgb

    largest_face = max(faces, key=lambda r: r[2] * r[3])
    return [largest_face], img_rgb

def draw_faces(image, faces, player_name):
    if image is None or faces is None:
        return image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        font_scale = max(0.5, min(1.5, 30 / len(player_name)))
        cv2.putText(image, player_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
    return image

def get_player_info(image_path):
    try:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """You are an expert AI trained in recognizing professional football (soccer) players with high accuracy. 
    Carefully analyze the given image and identify the player with certainty. 
    If a well-known player is present, return the following details in a structured format:

    - **Full Name**: (Only the player's full name, no extra text)
    - **Nationality**: (Country of origin)
    - **Club**: (Current club name)
    - **Position**: (Primary playing position)
    - **Achievements**: (Notable titles or awards, max 2-3)

    If multiple players are detected, return only the **most famous one**.
    If no player is recognized, respond with "Unknown".
"""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        }
                    ]
                }
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return "Unknown"
    except Exception as e:
        print(f"Error in get_player_info: {e}")
        return "Unknown"

@app.get("/")
async def get_form():
    return JSONResponse(content={"message": "Upload an image to identify a football player", "status": "ready"})

@app.post("/")
async def process_image(file: UploadFile = File(None)):
    if not file:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

    try:
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        faces, img_rgb = detect_faces(image_path)

        if faces and img_rgb is not None:
            player_info = get_player_info(image_path)
            player_name = player_info.split("\n")[0].replace("**Full Name**: ", "").strip() if player_info != "Unknown" else "Unknown"
            processed_img = draw_faces(img_rgb, faces, player_name)

            result_img_filename = f"result_{file.filename}"
            result_img_path = os.path.join(UPLOAD_FOLDER, result_img_filename)
            cv2.imwrite(result_img_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))

            return JSONResponse(content={
                "player_info": player_info,
                "result_image_url": f"/static/Uploads/{result_img_filename}",
                "status": "success"
            })
        else:
            return JSONResponse(content={
                "player_info": "No faces detected. Try another image.",
                "result_image_url": None,
                "status": "partial"
            })
    except Exception as e:
        print(f"Error processing image: {e}")
        return JSONResponse(content={
            "error": "Error processing image. Please try again.",
            "status": "error"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000)