from fastapi import FastAPI, Form
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
import re
from dotenv import load_dotenv
import os
from pydantic import BaseModel

app = FastAPI()

load_dotenv()
llm_resto = ChatGroq(
    api_key=os.getenv("recommendationKey"),
    model="llama-3.3-70b-versatile",
    temperature=0.0
)

prompt_template_resto = PromptTemplate(
    input_variables=['age', 'gender', 'weight', 'height', 'veg_or_nonveg', 'disease', 'region', 'allergics', 'foodtype'],
    template=(
        "Diet Recommendation System:\n"
        "I want you to provide output in the following format using the input criteria:\n\n"
        "Restaurants:\n"
        "- name1\n- name2\n- name3\n- name4\n- name5\n- name6\n\n"
        "Breakfast:\n"
        "- item1\n- item2\n- item3\n- item4\n- item5\n- item6\n\n"
        "Dinner:\n"
        "- item1\n- item2\n- item3\n- item4\n- item5\n\n"
        "Workouts:\n"
        "- workout1\n- workout2\n- workout3\n- workout4\n- workout5\n- workout6\n\n"
        "Criteria:\n"
        "Age: {age}, Gender: {gender}, Weight: {weight} kg, Height: {height} ft, "
        "Vegetarian: {veg_or_nonveg}, Disease: {disease}, Region: {region}, "
        "Allergics: {allergics}, Food Preference: {foodtype}.\n"
    )
)

class DietInput(BaseModel):
    age: str
    gender: str
    weight: str
    height: str
    veg_or_nonveg: str
    disease: str
    region: str
    allergics: str
    foodtype: str

@app.post("/recommend")
async def recommend(input_data: DietInput):
    chain = LLMChain(llm=llm_resto, prompt=prompt_template_resto)

    input_dict = input_data.dict()

    results = chain.run(input_dict)

    restaurant_names = re.findall(r'Restaurants:\s*(.*?)\n\n', results, re.DOTALL)
    breakfast_names = re.findall(r'Breakfast:\s*(.*?)\n\n', results, re.DOTALL)
    dinner_names = re.findall(r'Dinner:\s*(.*?)\n\n', results, re.DOTALL)
    workout_names = re.findall(r'Workouts:\s*(.*?)\n\n', results, re.DOTALL)

    def clean_list(block):
        return [line.strip("- ") for line in block.strip().split("\n") if line.strip()]

    response = {
        "restaurants": clean_list(restaurant_names[0]) if restaurant_names else [],
        "breakfast": clean_list(breakfast_names[0]) if breakfast_names else [],
        "dinner": clean_list(dinner_names[0]) if dinner_names else [],
        "workouts": clean_list(workout_names[0]) if workout_names else []
    }

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)