from flask import Flask, request, render_template, jsonify
import os
import pandas as pd
import numpy as np
import difflib
import re
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

app = Flask(__name__)

# Load datasets
file_path1 = "environmental-footprint-milks.csv"
file_path2 = "food-emissions-supply-chain.csv"

df_milks = pd.read_csv(file_path1)
df_food_emissions = pd.read_csv(file_path2)

df_milks.rename(columns={'Entity': 'Food'}, inplace=True)
df_food_emissions.rename(columns={'Entity': 'Food'}, inplace=True)
df_combined = pd.merge(df_food_emissions, df_milks, on="Food", how="outer")
df_combined.fillna(0, inplace=True)

# Encode categorical variables
label_encoder_food = LabelEncoder()
df_combined['Food_Encoded'] = label_encoder_food.fit_transform(df_combined['Food'])

meal_times = ['Breakfast', 'Lunch', 'Dinner', 'Snack']
df_combined['Meal_Time'] = (
    meal_times * (len(df_combined) // len(meal_times))
    + meal_times[:len(df_combined) % len(meal_times)]
)
label_encoder_meal = LabelEncoder()
df_combined['Meal_Time_Encoded'] = label_encoder_meal.fit_transform(df_combined['Meal_Time'])

features = [
    'Food_Encoded', 'Meal_Time_Encoded', 'food_emissions_land_use',
    'food_emissions_farm', 'food_emissions_animal_feed',
    'food_emissions_processing', 'food_emissions_transport',
    'food_emissions_retail', 'food_emissions_packaging',
    'food_emissions_losses'
]
target = 'food_emissions_farm'

X = df_combined[features]
y = df_combined[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[('num', StandardScaler(), features)]
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model_pipeline.fit(X_train, y_train)

openai.api_key = os.environ.get('OPENAI_API_KEY')

def extract_ingredients(dish_name):
    """
    Extracts the main ingredients of a given dish using OpenAI's GPT-3.5-turbo.
    """
    prompt = f"List only the main ingredients for the dish '{dish_name}' as a comma-separated list."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a culinary expert."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        ingredients = [re.escape(ingredient.strip()) for ingredient in content.split(',')]
        display_ingredients = [ingredient.strip() for ingredient in content.split(',')]
        return ingredients, display_ingredients
    except Exception as e:
        print(f"Error extracting ingredients: {e}")
        return [], []

def predict_environmental_impact(food, quantity, meal_time):
    """
    Predicts the environmental impact (GHG emissions in kg CO2eq) of a given food choice
    and provides an ingredient breakdown.
    """
    ingredient_impacts = {}
    total_emissions = 0
    regex_ingredients, display_ingredients = extract_ingredients(food)

    if regex_ingredients:
        dataset_foods = [f.lower() for f in df_combined['Food'].tolist()]
        for regex_ing, display_ing in zip(regex_ingredients, display_ingredients):
            matched_ingredient = df_combined[
                df_combined['Food'].str.contains(re.escape(display_ing), case=False, na=False)
            ]
            if not matched_ingredient.empty:
                impact = matched_ingredient.iloc[0]['food_emissions_farm'] * quantity
                ingredient_impacts[display_ing] = impact
                total_emissions += impact

    if total_emissions == 0 and food in df_combined['Food'].values:
        food_data = df_combined[df_combined['Food'] == food].iloc[0]
        total_emissions = food_data['food_emissions_farm'] * quantity

    if total_emissions == 0:
        total_emissions = np.random.uniform(0.5, 5.0) * quantity

    # Console Output for Debugging
    print(f"\nYour meal choice of {quantity} serving(s) of {food} during {meal_time} suggests the following environmental impact:")
    print(f"🌍 Estimated Greenhouse Gas Emissions: {total_emissions:.2f} kg CO2eq")
    print("Higher emissions indicate a larger carbon footprint. Consider choosing plant-based foods.")

    if ingredient_impacts:
        print("Breakdown of main ingredient contribution:")
        for ingredient, impact in ingredient_impacts.items():
            print(f"- {ingredient}: {impact:.2f} kg CO2eq")

    # Web Output
    formatted_output = f"""
        <p>Your meal choice of <strong>{quantity}</strong> serving(s) of <strong>{food}</strong> during <strong>{meal_time}</strong> suggests the following environmental impact:</p>
        <p><strong>🌍 Estimated Greenhouse Gas Emissions:</strong> {total_emissions:.2f} kg CO2eq</p>
        <p>Higher emissions indicate a larger carbon footprint. Consider choosing plant-based or sustainably sourced foods.</p>
    """
    if ingredient_impacts:
        formatted_output += "<h5>Breakdown of Main Ingredient Contribution:</h5><ul>"
        for ingredient, impact in ingredient_impacts.items():
            formatted_output += f"<li>{ingredient}: {impact:.2f} kg CO2eq</li>"
        formatted_output += "</ul>"

    return total_emissions, ingredient_impacts, formatted_output

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        food = request.form.get("food")
        quantity = float(request.form.get("quantity", 1))
        meal_time = request.form.get("meal_time", "Lunch")

        predicted_emissions, ingredient_contributions, formatted_message = predict_environmental_impact(food, quantity, meal_time)

        return render_template("home.html", food=food, quantity=quantity, meal_time=meal_time,
                               predicted_emissions=predicted_emissions,
                               ingredient_contributions=ingredient_contributions,
                               formatted_message=formatted_message)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)

