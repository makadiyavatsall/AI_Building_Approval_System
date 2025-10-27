import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import os

print("🚀 Script started...")

# ====== Load Dataset ======
df = pd.read_csv('municipal_building_plan_dataset_10000.csv')
print(f"✅ Dataset loaded successfully! Rows: {len(df)}, Columns: {len(df.columns)}")

# ====== Encode Categorical Columns ======
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("✅ Encoding complete.")

# ====== Split Dataset ======
X = df.drop('Approval_Status', axis=1)
y = df['Approval_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("✅ Data split complete.")

# ====== Scale Features ======
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Feature scaling complete.")

# ====== Define Models ======
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric='logloss', n_jobs=1, verbosity=1)
}

results = []

# ====== Train and Evaluate Models ======
for name, model in models.items():
    print(f"\n🔹 Training {name} ...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append((name, acc))
    print(f"✅ {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

print("\n✅ Model training loop completed!")

# ====== Select Best Model ======
best_model_name, best_acc = max(results, key=lambda x: x[1])
best_model = models[best_model_name]
print(f"\n🏆 Best Model: {best_model_name} | Accuracy: {best_acc:.4f}")

# ====== Ensure output directory exists ======
output_dir = "../Model_Prediction"
os.makedirs(output_dir, exist_ok=True)

# ====== Save Artifacts ======
joblib.dump(best_model, f"{output_dir}/best_model.pkl")
joblib.dump(scaler, f"{output_dir}/scaler.pkl")
joblib.dump(label_encoders, f"{output_dir}/label_encoder.pkl")

with open(f"{output_dir}/feature_columns.txt", "w") as f:
    f.write("\n".join(X.columns))

pd.DataFrame(results, columns=["Model", "Accuracy"]).to_csv("model_performance_results.csv", index=False)

print("\n✅ Training complete and model artifacts saved successfully!")
print(f"📁 Saved to: {os.path.abspath(output_dir)}")
