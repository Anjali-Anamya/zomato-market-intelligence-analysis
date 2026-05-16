# Zomato Market Intelligence App

End-to-end Data Science pipeline — Restaurant Success Prediction & Area Market Zone Clustering.

Built on 8,923 Zomato restaurant records from Bangalore.

## Features

- **Restaurant Success Predictor** — Random Forest classifier predicts if a restaurant will be high-rated (≥4.0). Accuracy: 88.4%, F1: 0.775
- **Area Market Zone Classifier** — K-Means clustering (K=4) segments 147 Bangalore areas into Growth Opportunity, Saturated, Underserved, and High Competition Low Demand zones
- **Feature Importance chart** — shows which inputs drive restaurant success most

## Local Setup

1. Clone this repo
2. Place `zomato_feature_engineered.csv` in the root folder
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Train and save models:
   ```
   python train_models.py
   ```
5. Run the app:
   ```
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Push this repo (including the `models/` folder) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select repo → branch → `app.py` as the main file
5. Click **Deploy** — done!

> Note: Make sure the `models/` folder with `.pkl` files is committed to GitHub before deploying.

## Project Structure

```
zomato_app/
├── app.py               # Main Streamlit application
├── train_models.py      # Script to train and save models
├── requirements.txt     # Python dependencies
├── models/
│   ├── rf_model.pkl     # Trained Random Forest
│   ├── kmeans_model.pkl # Trained K-Means
│   ├── scaler.pkl       # StandardScaler for clustering
│   └── area_df.pkl      # Area-level aggregated data with cluster labels
└── README.md
```

## Tech Stack

- Python, scikit-learn, pandas, NumPy
- Streamlit (frontend + deployment)
- Matplotlib (visualizations)
- PostgreSQL (data analytics — separate pipeline)
- Power BI (business dashboards — separate pipeline)
