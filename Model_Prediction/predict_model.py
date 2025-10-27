import pandas as pd
import joblib
import numpy as np
import os

print("🚀 Starting prediction pipeline...")

# ====== Load Artifacts ======
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoder.pkl")

with open("feature_columns.txt", "r") as f:
    feature_columns = f.read().splitlines()

print("✅ Model and preprocessors loaded successfully!")

# ====== Function to preprocess and predict ======
def predict_building_approval(new_data_dict):
    """
    new_data_dict: Dictionary containing feature_name -> value
    Example:
        {
            "Building_Type": "Residential",
            "Zone": "Urban",
            "Height": 25,
            "Floors": 5,
            "Owner_Type": "Private",
            ...
        }
    """
    df = pd.DataFrame([new_data_dict])

    # Handle missing columns
    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    df = df[feature_columns]

    # Apply label encoding to categorical columns
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                df[col] = le.transform(df[col])
            except ValueError:
                # Handle unseen categories
                df[col] = df[col].map(lambda x: le.classes_[0] if x not in le.classes_ else x)
                df[col] = le.transform(df[col])

    # Fill any missing values (if unseen)
    df.fillna(0, inplace=True)

    # Scale numerical columns
    X_scaled = scaler.transform(df)

    # Predict
    prediction = model.predict(X_scaled)[0]
    result = "✅ Approved" if prediction == 1 else "❌ Not Approved"
    return result


# ====== Example Usage ======
if __name__ == "__main__":
    # 🔹 Example new data (edit this for your test)
    new_application = {
        "Building_Type": "Residential",
        "Zone": "Urban",
        "Height": 20,
        "Floors": 4,
        "Owner_Type": "Private",
        "Land_Area": 2500,
        "Parking_Area": 300,
        "Green_Cover": 25,
        "Fire_Safety_Compliance": "Yes",
        "Earthquake_Resistance": "Yes",
        "Road_Access_Width": 10,
        "Drainage_System": "Yes",
        "Water_Supply": "Yes",
        "Electricity_Connection": "Yes"
    }

    print("🔍 Predicting approval for new application...")
    result = predict_building_approval(new_application)
    print(f"🏗️ Prediction Result: {result}")
