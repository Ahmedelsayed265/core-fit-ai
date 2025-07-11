from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import random
from math import ceil
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import List, Optional

app = FastAPI()

# Load datasets
BASE_DIR = os.path.dirname(__file__)
fitness_exercise_df = pd.read_csv(os.path.join(BASE_DIR, "fitness_exercises.csv"))
food_df = pd.read_csv(os.path.join(BASE_DIR, "food_data.csv"))

# Prepare data for AI model
def prepare_food_data():
    X = food_df[['Caloric Value', 'Protein', 'Carbohydrates', 'Fat']].values
    y = (food_df['Caloric Value'] > food_df['Caloric Value'].median()).astype(int)
    return X, y

# Train a simple AI model
X, y = prepare_food_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
    if gender.lower() == "male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

def calculate_tdee(bmr: float, activity_level: str) -> float:
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very_active": 1.9
    }
    return bmr * activity_multipliers.get(activity_level.lower(), 1.2)

def adjust_calories_for_goal(tdee: float, goal: str) -> float:
    if goal.lower() == "lose_weight":
        return tdee - 500
    elif goal.lower() == "gain_muscle":
        return tdee + 500
    return tdee

def generate_ai_meal_plan(calories: float, preferred_foods: List[str], disliked_foods: List[str], allergies: List[str]) -> List[dict]:
    filtered_foods = food_df[
        (~food_df["food"].isin(disliked_foods)) &
        (~food_df["food"].isin(allergies))
    ]
    if preferred_foods:
        filtered_foods = filtered_foods[filtered_foods["food"].isin(preferred_foods) | ~filtered_foods["food"].isin(preferred_foods)]

    meal_calories = {
        "breakfast": calories * 0.25,
        "lunch": calories * 0.30,
        "dinner": calories * 0.30,
        "snack": calories * 0.15
    }

    meal_plan = []
    for _ in range(7):
        daily_meals = {}
        for meal_type, meal_cals in meal_calories.items():
            selected_foods = []
            total_cals = 0
            attempts = 0
            max_attempts = 10
            food_features = filtered_foods[['Caloric Value', 'Protein', 'Carbohydrates', 'Fat']].values
            food_features_scaled = scaler.transform(food_features)
            predictions = model.predict_proba(food_features_scaled)
            suitable_indices = (predictions[:, 1] > 0.5) & (filtered_foods['Caloric Value'] <= meal_cals * 1.1)
            suitable_foods = filtered_foods[suitable_indices]
            while total_cals < meal_cals * 0.9 and attempts < max_attempts and not suitable_foods.empty:
                food = suitable_foods.sample(1).iloc[0]
                food_cals = float(food["Caloric Value"])
                if total_cals + food_cals <= meal_cals * 1.1:
                    selected_foods.append({
                        "food": food["food"],
                        "calories": food_cals,
                        "protein": float(food["Protein"]),
                        "carbs": float(food["Carbohydrates"]),
                        "fat": float(food["Fat"])
                    })
                    total_cals += food_cals
                attempts += 1
            daily_meals[meal_type] = selected_foods
        meal_plan.append(daily_meals)
    return meal_plan

def generate_exercise_plan(goal: str, fitness_level: str, exercise_days: int) -> List[List[dict]]:
    if goal.lower() == "lose_weight":
        target_parts = ["cardio", "waist", "upper legs"]
    elif goal.lower() == "gain_muscle":
        target_parts = ["chest", "back", "upper arms", "shoulders", "upper legs"]
    else:
        target_parts = fitness_exercise_df["bodyPart"].unique().tolist()

    filtered_exercises = fitness_exercise_df[fitness_exercise_df["bodyPart"].isin(target_parts)]

    if fitness_level.lower() == "beginner":
        sets = 2
        reps = "8-12"
    elif fitness_level.lower() == "intermediate":
        sets = 3
        reps = "10-15"
    else:
        sets = 4
        reps = "12-15"

    exercise_plan = []
    for _ in range(exercise_days):
        daily_exercises = filtered_exercises.sample(4).to_dict("records")
        exercise_plan.append([
            {
                "name": ex["name"],
                "bodyPart": ex["bodyPart"],
                "sets": sets,
                "reps": reps
            } for ex in daily_exercises
        ])
    return exercise_plan

class UserData(BaseModel):
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str
    goal: str
    fitness_level: str
    exercise_days: int
    preferred_foods: Optional[List[str]] = None
    disliked_foods: Optional[List[str]] = None
    allergies: Optional[List[str]] = None

@app.get("/")
async def root():
    return {"message": "Welcome to the Fitness and Meal Planning API"}

@app.post("/plan")
async def get_plan(user_data: UserData):
    try:
        bmr = calculate_bmr(user_data.weight, user_data.height, user_data.age, user_data.gender)
        tdee = calculate_tdee(bmr, user_data.activity_level)
        daily_calories = adjust_calories_for_goal(tdee, user_data.goal)

        meal_plan = generate_ai_meal_plan(
            daily_calories,
            user_data.preferred_foods or [],
            user_data.disliked_foods or [],
            user_data.allergies or []
        )
        exercise_plan = generate_exercise_plan(
            user_data.goal,
            user_data.fitness_level,
            user_data.exercise_days
        )

        return {
            "bmr": bmr,
            "tdee": tdee,
            "daily_calories": daily_calories,
            "meal_plan": meal_plan,
            "exercise_plan": exercise_plan
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))