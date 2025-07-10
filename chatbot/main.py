# from fastapi import FastAPI, Form, Request, Response, Cookie
# from fastapi.responses import JSONResponse, RedirectResponse
# from fastapi.middleware.cors import CORSMiddleware
# import google.generativeai as genai
# import json
# import os
# import hashlib

# GOOGLE_API_KEY = "AIzaSyC96jCvKWFF2Nghq2ZvWSzU-QpgbK_vy-s"
# genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel('gemini-1.5-flash')

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # للسهولة
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# USERS_FILE = "users.json"

# def load_users():
#     if os.path.exists(USERS_FILE):
#         with open(USERS_FILE, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {}

# def save_users(users):
#     with open(USERS_FILE, "w", encoding="utf-8") as f:
#         json.dump(users, f, indent=2, ensure_ascii=False)

# def hash_password(password):
#     return hashlib.sha256(password.encode()).hexdigest()

# def is_authenticated(username, password):
#     users = load_users()
#     hashed = hash_password(password)
#     return username in users and users[username]["password"] == hashed

# def load_user_history(username):
#     filename = f"chat_memory_{username}.json"
#     if os.path.exists(filename):
#         with open(filename, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return []

# def save_user_history(username, history):
#     filename = f"chat_memory_{username}.json"
#     with open(filename, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2, ensure_ascii=False)

# @app.post("/register")
# async def register(username: str = Form(...), password: str = Form(...)):
#     users = load_users()
#     if username in users:
#         return JSONResponse({"message": "User already exists"}, status_code=400)
#     users[username] = {"password": hash_password(password)}
#     save_users(users)
#     return JSONResponse({"message": "Registered successfully"})

# @app.post("/login")
# async def login(response: Response, username: str = Form(...), password: str = Form(...)):
#     if is_authenticated(username, password):
#         resp = JSONResponse({"message": "Login successful"})
#         resp.set_cookie(key="username", value=username, httponly=True)
#         return resp
#     return JSONResponse({"message": "Invalid username or password"}, status_code=401)

# @app.post("/ask")
# async def ask(username: str = Form(...), message: str = Form(...)):
#     chat_history = load_user_history(username)
#     chat_history.append({
#         "role": "user",
#         "parts": [{"text": message}]
#     })
#     try:
#         response = model.generate_content(chat_history)
#         reply = response.text.strip()

#         chat_history.append({
#             "role": "model",
#             "parts": [{"text": reply}]
#         })
#         save_user_history(username, chat_history)
#         return JSONResponse({"reply": reply})
#     except Exception as e:
#         return JSONResponse({"reply": f"Error: {str(e)}"})

# @app.get("/history")
# async def get_history(username: str = Cookie(default=None)):
#     if not username:
#         return JSONResponse({"message": "Unauthorized"}, status_code=401)
#     history = load_user_history(username)
#     return JSONResponse(history)

# @app.get("/logout")
# async def logout():
#     resp = JSONResponse({"message": "Logged out"})
#     resp.delete_cookie("username")
#     return resp


from fastapi import FastAPI, Form, Request, Response, Cookie
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
import os
import hashlib

GOOGLE_API_KEY = "AIzaSyC96jCvKWFF2Nghq2ZvWSzU-QpgbK_vy-s"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_authenticated(username, password):
    users = load_users()
    hashed = hash_password(password)
    return username in users and users[username]["password"] == hashed

def load_user_history(username):
    filename = f"chat_memory_{username}.json"
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_user_history(username, history):
    filename = f"chat_memory_{username}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...)):
    users = load_users()
    if username in users:
        return JSONResponse({"message": "User already exists"}, status_code=400)
    users[username] = {"password": hash_password(password)}
    save_users(users)
    return JSONResponse({"message": "Registered successfully"})

@app.post("/login")
async def login(response: Response, username: str = Form(...), password: str = Form(...)):
    if is_authenticated(username, password):
        resp = JSONResponse({"message": "Login successful"})
        resp.set_cookie(key="username", value=username, httponly=True)
        return resp
    return JSONResponse({"message": "Invalid username or password"}, status_code=401)

@app.post("/ask")
async def ask(request: Request, message: str = Form(...), username: str = Cookie(default=None)):
    if not username:
        return JSONResponse({"message": "Unauthorized"}, status_code=401)
    chat_history = load_user_history(username)
    chat_history.append({
        "role": "user",
        "parts": [{"text": message}]
    })
    try:
        response = model.generate_content(chat_history)
        reply = response.text.strip()
        chat_history.append({
            "role": "model",
            "parts": [{"text": reply}]
        })
        save_user_history(username, chat_history)
        return JSONResponse({"reply": reply})
    except Exception as e:
        return JSONResponse({"reply": f"Error: {str(e)}"})

@app.get("/history")
async def get_history(username: str = Cookie(default=None)):
    if not username:
        return JSONResponse({"message": "Unauthorized"}, status_code=401)
    history = load_user_history(username)
    return JSONResponse(history)

@app.get("/logout")
async def logout():
    resp = JSONResponse({"message": "Logged out"})
    resp.delete_cookie("username")
    return resp