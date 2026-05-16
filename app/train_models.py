"""
Run this script once locally to train and save the models.
Place your zomato_feature_engineered.csv in the same folder before running.

Usage:
    python train_models.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

df = pd.read_csv("zomato_feature_engineered.csv")

FEATURES = [
    'dinner_ratings', 'dinner_reviews', 'delivery_reviews', 'averagecost',
    'ishomedelivery', 'istakeaway', 'isindoorseating', 'isvegonly',
    'cuisine_count', 'restaurant_density', 'cost_index',
    'demand_score', 'opportunity_score', 'area_avg_rating'
]

df = df.dropna(subset=FEATURES + ['overall_rating'])
df['success'] = (df['overall_rating'] >= 4.0).astype(int)

X = df[FEATURES]
y = df['success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.3f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.3f}")

print("Training K-Means clustering...")
AREA_FEATURES = ['area_avg_rating', 'restaurant_density', 'demand_score', 'opportunity_score']
area_df = df.groupby('area')[AREA_FEATURES].mean().reset_index().dropna()
scaler = StandardScaler()
X_cluster = scaler.fit_transform(area_df[AREA_FEATURES])
km = KMeans(n_clusters=4, random_state=42, n_init=10)
area_df['cluster'] = km.fit_predict(X_cluster)
print(f"  Areas segmented: {len(area_df)}")

os.makedirs("models", exist_ok=True)
joblib.dump(rf,      "models/rf_model.pkl")
joblib.dump(km,      "models/kmeans_model.pkl")
joblib.dump(scaler,  "models/scaler.pkl")
joblib.dump(area_df, "models/area_df.pkl")

print("\nAll models saved to /models folder.")
print("You can now run: streamlit run app.py")
