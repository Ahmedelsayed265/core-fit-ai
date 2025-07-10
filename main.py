from fastapi import FastAPI
from chatbot.main import app as chatbot_app
# from footballPlayerDetection.main import app as football_app
from recommendation.main import app as recommendation_app
from weaklyRecommendation.main import app as weakly_recommendation_app

app = FastAPI()

app.mount("/chatbot", chatbot_app)
# app.mount("/footballPlayerDetection", football_app)
app.mount("/recommendation", recommendation_app)
app.mount("/weaklyRecommendation", weakly_recommendation_app)
