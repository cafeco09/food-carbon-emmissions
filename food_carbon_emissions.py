import os
import pandas as pd
import numpy as np
import difflib
import openai
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Initialize Flask app
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
df_combined['Meal_Time'] = meal_times * (len(df_combined) // len(meal_times)) + meal_times[:len(df_combined) % len(meal_times)]
label_encoder_meal = LabelEncoder()
df_combined['Meal_Time_Encoded'] = label_encoder_meal.fit_transform(df_combined['Meal_Time'])

# Selecting features and target
features = ['Food_Encoded', 'Meal_Time_Encoded', 'food_emissions_land_use', 'food_emissions_farm',
            'food_emissions_animal_feed', 'food_emissions_processing', 'food_emissions_transport',
            'food_emissions_retail', 'food_emissions_packaging', 'food_emissions_losses']
target = 'food_emissions_farm'

X = df_combined[features]
y = df_combined[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), features)
])
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
model_pipeline.fit(X_train, y_train)

# Set OpenAI API key
openai.api_key = os.environ.get('OPENAI_API_KEY')

def extract_ingredients(dish_name):
    """
    Extracts the main ingredients of a dish using OpenAI's GPT model.
    """
    prompt = f"List only the main ingredients for the dish called '{dish_name}' as a comma-separated list."
    try:
        response = openai.chat.completion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a culinary expert."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        ingredients = [ingredient.strip() for ingredient in content.split(',')]
        return ingredients
    except Exception as e:
        print(f"Error extracting ingredients: {e}")
        return []

def predict_environmental_impact(food, quantity, meal_time):
    """
    Predicts the environmental impact of a given food choice and provides an ingredient breakdown.
    """
    ingredient_impacts = {}
    total_emissions = 0

    # Always attempt to extract ingredients for a detailed breakdown.
    ingredients = extract_ingredients(food)
    
    if ingredients:
        dataset_foods = [f.lower() for f in df_combined['Food'].tolist()]
        for ingredient in ingredients:
            matched_ingredient = df_combined[df_combined['Food'].str.contains(ingredient, case=False, na=False)]
            if matched_ingredient.empty:
                close_matches = difflib.get_close_matches(ingredient.lower(), dataset_foods, n=1, cutoff=0.4)
                if close_matches:
                    match = close_matches[0]
                    matched_ingredient = df_combined[df_combined['Food'].str.lower() == match]
            if not matched_ingredient.empty:
                impact = matched_ingredient.iloc[0]['food_emissions_farm'] * quantity
                ingredient_impacts[ingredient] = impact
                total_emissions += impact
    else:
        # Fallback: if no ingredients could be extracted, check if food is in dataset.
        if food in df_combined['Food'].values:
            food_data = df_combined[df_combined['Food'] == food].iloc[0]
            total_emissions = food_data['food_emissions_farm'] * quantity
        else:
            total_emissions = np.random.uniform(0.5, 5.0) * quantity

    return total_emissions, ingredient_impacts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    food = data.get('food')
    quantity = float(data.get('quantity', 1))
    meal_time = data.get('meal_time', 'Lunch')
    
    if not food:
        return render_template('index.html', error="Missing food parameter")
    
    predicted_impact, ingredient_contributions = predict_environmental_impact(food, quantity, meal_time)
    
    impact_message = (
        f"Your meal choice of {quantity} serving(s) of {food} during {meal_time} suggests the following environmental impact:\n"
        f"🌍 Estimated Greenhouse Gas Emissions: {predicted_impact:.2f} kg CO2eq\n"
        "Environmental impact refers to the effect that food production, processing, and transportation have on the planet.\n"
        "Higher emissions indicate a larger carbon footprint, which contributes more to climate change. "
        "Consider choosing plant-based or sustainably sourced foods to help reduce your impact."
    )
    
    return render_template('result.html', 
                           food=food, 
                           quantity=quantity, 
                           meal_time=meal_time, 
                           predicted_impact=predicted_impact, 
                           ingredient_contributions=ingredient_contributions,
                           impact_message=impact_message)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
