import os
import pandas as pd
import numpy as np
import difflib
import openai
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from flask import Flask

# Load datasets
file_path1 = "environmental-footprint-milks.csv"
file_path2 = "food-emissions-supply-chain.csv"

df_milks = pd.read_csv(file_path1)
df_food_emissions = pd.read_csv(file_path2)

# Merge datasets
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

# Selecting features and target
features = [
    'Food_Encoded', 'Meal_Time_Encoded', 'food_emissions_land_use',
    'food_emissions_farm', 'food_emissions_animal_feed',
    'food_emissions_processing', 'food_emissions_transport',
    'food_emissions_retail', 'food_emissions_packaging',
    'food_emissions_losses'
]
target = 'food_emissions_farm'  # Using farm emissions as a proxy for environmental impact

X = df_combined[features]
y = df_combined[target]

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creating a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), features)
    ]
)

# Defining the model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Training the model
model_pipeline.fit(X_train, y_train)

# Retrieve OpenAI API key from environment variable (e.g., set via GitHub Secrets)
openai.api_key = os.environ.get('OPENAI_API_KEY')
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")


def extract_ingredients(dish_name):
    """
    Extracts the main ingredients of a given dish using OpenAI's GPT-3.5-turbo model.
    
    Args:
        dish_name (str): The name of the dish.
    
    Returns:
        list: A list of extracted ingredients.
    """
    prompt = (
        f"List only the main ingredients for the dish called '{dish_name}' as a "
        "comma-separated list with no extra commentary."
    )
    try:
        response = openai.chat.completions.create(
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
        print(f"An error occurred while extracting ingredients: {e}")
        return []


def predict_environmental_impact(food, quantity, meal_time):
    """
    Predicts the environmental impact (GHG emissions in kg CO2eq) of a given food choice
    and provides an ingredient breakdown.

    It always attempts to extract ingredients and uses fuzzy matching to find the best
    dataset match for each ingredient. If no ingredient contributions are found, it falls
    back to using an exact match (if available) or a random estimation.

    Returns:
        tuple: total emissions and a breakdown of emissions per ingredient.
    """
    ingredient_impacts = {}
    total_emissions = 0

    # Always attempt to extract ingredients for a detailed breakdown.
    ingredients = extract_ingredients(food)
    
    if ingredients:
        dataset_foods = [f.lower() for f in df_combined['Food'].tolist()]
        for ingredient in ingredients:
            # First, try direct substring search.
            matched_ingredient = df_combined[
                df_combined['Food'].str.contains(ingredient, case=False, na=False)
            ]
            
            # If no direct match, try token matching.
            if matched_ingredient.empty:
                tokens = ingredient.split()
                for token in tokens:
                    token = token.strip().lower()
                    matched_ingredient = df_combined[
                        df_combined['Food'].str.lower().str.contains(token, na=False)
                    ]
                    if not matched_ingredient.empty:
                        break
            
            # If still no match, use fuzzy matching on tokens.
            if matched_ingredient.empty:
                tokens = ingredient.split()
                for token in tokens:
                    token = token.strip().lower()
                    close_matches = difflib.get_close_matches(token, dataset_foods, n=1, cutoff=0.4)
                    if close_matches:
                        match = close_matches[0]
                        matched_ingredient = df_combined[
                            df_combined['Food'].str.lower() == match
                        ]
                        if not matched_ingredient.empty:
                            break
            
            if not matched_ingredient.empty:
                impact = matched_ingredient.iloc[0]['food_emissions_farm'] * quantity
                ingredient_impacts[ingredient] = impact
                total_emissions += impact

    # Fallback: if no ingredient breakdown was found, try an exact match.
    if total_emissions == 0 and food in df_combined['Food'].values:
        food_data = df_combined[df_combined['Food'] == food].iloc[0]
        total_emissions = food_data['food_emissions_farm'] * quantity

    # Final fallback: if still zero, use a random estimation.
    if total_emissions == 0:
        total_emissions = np.random.uniform(0.5, 5.0) * quantity

    return total_emissions, ingredient_impacts


# Create the Flask app and define routes
app = Flask(__name__)

@app.route('/')
def home():
    return "Food Carbon Emissions API is running!"


if __name__ == "__main__":
    # Define dish details for testing
    dish = "Pasta"
    quantity = 2
    meal_time = 'Lunch'
    
    # Extract and print ingredients for the dish
    ingredients = extract_ingredients(dish)
    print(f"Ingredients for {dish}: {ingredients}")

    # Predict environmental impact for the dish using improved matching on ingredients
    predicted_impact, ingredient_contributions = predict_environmental_impact(
        dish, quantity, meal_time
    )
    
    # Create and display a formatted overall impact message
    impact_message = (
        f"Your meal choice of {quantity} serving(s) of {dish} during {meal_time} suggests the following environmental impact:\n"
        f"🌍 Estimated Greenhouse Gas Emissions: {predicted_impact:.2f} kg CO2eq\n"
        "Environmental impact refers to the effect that food production, processing, and transportation have on the planet.\n"
        "Higher emissions indicate a larger carbon footprint, which contributes more to climate change. "
        "Consider choosing plant-based or sustainably sourced foods to help reduce your impact.\n"
    )
    print("\n" + impact_message)
    
    # Display the ingredient breakdown if available
    if ingredient_contributions:
        print("Breakdown of main ingredient contribution:")
        for ingredient, impact in ingredient_contributions.items():
            print(f"- {ingredient}: {impact:.2f} kg CO2eq")
    else:
        print("No detailed ingredient breakdown available.")

    # Start the Flask app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
