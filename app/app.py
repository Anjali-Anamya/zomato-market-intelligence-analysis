import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Zomato Market Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.metric-card {
    background: white; border: 1px solid #EBEBEB;
    border-radius: 12px; padding: 1.2rem 1.4rem; text-align: center;
}
.metric-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.metric-card .value { font-size: 28px; font-weight: 600; color: #1A1A1A; }
.metric-card .sub   { font-size: 12px; color: #aaa; margin-top: 2px; }
.predict-box { padding: 1.5rem 2rem; border-radius: 16px; text-align: center; margin-top: 1rem; }
.success-box { background: #EDFAF3; border: 1.5px solid #34C77B; }
.fail-box    { background: #FEF2F2; border: 1.5px solid #F87171; }
</style>
""", unsafe_allow_html=True)

# ── Load models & data ────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf      = joblib.load("models/rf_model.pkl")
    km      = joblib.load("models/kmeans_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    area_df = joblib.load("models/area_df.pkl")
    return rf, km, scaler, area_df

@st.cache_data
def load_data():
    df = pd.read_csv("../data/zomato_feature_engineered.csv")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["delivery_ratings"] = pd.to_numeric(df["delivery_ratings"], errors="coerce")
    df["dinner_ratings"]   = pd.to_numeric(df["dinner_ratings"],   errors="coerce")
    df["high_rated"] = (df["delivery_ratings"] >= 4.0).astype(int)
    return df

rf, km, scaler, area_df = load_models()
df = load_data()

FEATURES = [
    'dinner_ratings','dinner_reviews','delivery_reviews','averagecost',
    'ishomedelivery','istakeaway','isindoorseating','isvegonly',
    'cuisine_count','restaurant_density','cost_index',
    'demand_score','opportunity_score','area_avg_rating'
]

ZONE_STYLE = {
    "Growth Opportunity":          ("#2ca02c", "#EDFAF3"),
    "Saturated Market":            ("#d62728", "#FEF2F2"),
    "Underserved Area":            ("#e6b800", "#FFFDE7"),
    "High Competition Low Demand": ("#ff7f0e", "#FFF3E0"),
}

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🍽️ Zomato Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Overview", "🤖 Predict Success", "🗺️ Area Clustering"])
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:12px; color:#999; line-height:1.7;'>
Built by <b>Anjali</b><br>
Random Forest · K-Means<br>
8,923 Zomato records · Bangalore
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Zomato Market Intelligence")
    st.markdown("##### End-to-end Data Science pipeline — Restaurant Success Prediction & Area Clustering")
    st.markdown("")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""<div class='metric-card'>
            <div class='label'>Restaurants</div>
            <div class='value'>8,923</div>
            <div class='sub'>Bangalore records</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class='metric-card'>
            <div class='label'>Model Accuracy</div>
            <div class='value'>64.6%</div>
            <div class='sub'>Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Areas Segmented</div>
            <div class='value'>{len(area_df)}</div>
            <div class='sub'>into 4 market zones</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""<div class='metric-card'>
            <div class='label'>Features Engineered</div>
            <div class='value'>8</div>
            <div class='sub'>custom features</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🤖 Restaurant Success Predictor")
        st.markdown("""
        Select any restaurant — values are **fetched automatically** from the dataset
        and fed into the model. Shows prediction vs actual result.
        - **Model**: Random Forest Classifier
        - **Features**: 14 engineered inputs
        - **Accuracy**: 64.6%
        """)
    with col2:
        st.markdown("#### 🗺️ Area Market Zone Classifier")
        st.markdown("""
        Select a Bangalore area to see its **market zone** fetched from the clustering results.
        - **Model**: K-Means Clustering (K=4)
        - **Zones**: Growth Opportunity · Saturated · Underserved · High Competition Low Demand
        """)

    st.markdown("---")
    st.markdown("#### 📊 Feature Importance — What drives restaurant success?")

    feat_imp = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d62728' if v > 0.1 else '#4C72B0' for v in feat_imp['Importance']]
    bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, height=0.6)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')
    ax.axvline(0.1, color='red', linestyle='--', alpha=0.4, label='High impact (>0.1)')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, feat_imp['Importance']):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#555')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Success  (fetches all values from CSV)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Success":
    st.title("Restaurant Success Predictor")
    st.markdown("Select a restaurant — all 14 feature values are **fetched automatically** from the dataset.")
    st.markdown("")

    valid_df = df[FEATURES + ['name', 'area', 'cuisines', 'high_rated', 'overall_rating']].dropna()
    restaurant_names = sorted(valid_df['name'].unique().tolist())
    selected = st.selectbox("🔍 Search or select a restaurant", restaurant_names)

    if selected:
        row = valid_df[valid_df['name'] == selected].iloc[0]

        st.markdown("#### 📋 Values fetched from dataset")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Area",           row['area'])
        c2.metric("Dinner Rating",  f"{row['dinner_ratings']:.1f}")
        c3.metric("Avg Cost",       f"₹{int(row['averagecost'])}")
        c4.metric("Cuisines",       int(row['cuisine_count']))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Demand Score",       f"{row['demand_score']:.0f}")
        c2.metric("Opportunity Score",  f"{row['opportunity_score']:.1f}")
        c3.metric("Area Avg Rating",    f"{row['area_avg_rating']:.2f}")
        c4.metric("Restaurant Density", f"{int(row['restaurant_density'])}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Home Delivery",  "Yes" if row['ishomedelivery']  else "No")
        c2.metric("Takeaway",       "Yes" if row['istakeaway']      else "No")
        c3.metric("Indoor Seating", "Yes" if row['isindoorseating'] else "No")
        c4.metric("Veg Only",       "Yes" if row['isvegonly']       else "No")

        st.markdown("---")

        # predict using fetched values
        input_data = pd.DataFrame([row[FEATURES].values], columns=FEATURES)
        pred       = rf.predict(input_data)[0]
        prob       = rf.predict_proba(input_data)[0]
        confidence = prob[pred] * 100
        actual     = int(row['high_rated'])
        actual_rating = row['overall_rating']

        col_pred, col_actual = st.columns(2)

        with col_pred:
            st.markdown("#### 🤖 Model Prediction")
            if pred == 1:
                st.markdown(f"""
                <div class='predict-box success-box'>
                    <div style='font-size:36px'>✅</div>
                    <div style='font-size:20px; font-weight:600; color:#16A34A; margin:8px 0'>High-Rated</div>
                    <div style='font-size:13px; color:#555'>Predicted rating ≥ 4.0</div>
                    <div style='font-size:24px; font-weight:600; color:#16A34A; margin-top:10px'>{confidence:.1f}% confident</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='predict-box fail-box'>
                    <div style='font-size:36px'>⚠️</div>
                    <div style='font-size:20px; font-weight:600; color:#DC2626; margin:8px 0'>Low-Rated</div>
                    <div style='font-size:13px; color:#555'>Predicted rating &lt; 4.0</div>
                    <div style='font-size:24px; font-weight:600; color:#DC2626; margin-top:10px'>{confidence:.1f}% confident</div>
                </div>""", unsafe_allow_html=True)

        with col_actual:
            st.markdown("#### 📊 Actual from Dataset")
            if actual == 1:
                st.markdown(f"""
                <div class='predict-box success-box'>
                    <div style='font-size:36px'>⭐</div>
                    <div style='font-size:20px; font-weight:600; color:#16A34A; margin:8px 0'>High-Rated</div>
                    <div style='font-size:13px; color:#555'>Actual rating ≥ 4.0</div>
                    <div style='font-size:24px; font-weight:600; color:#16A34A; margin-top:10px'>Rating: {actual_rating:.2f}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='predict-box fail-box'>
                    <div style='font-size:36px'>📉</div>
                    <div style='font-size:20px; font-weight:600; color:#DC2626; margin:8px 0'>Low-Rated</div>
                    <div style='font-size:13px; color:#555'>Actual rating &lt; 4.0</div>
                    <div style='font-size:24px; font-weight:600; color:#DC2626; margin-top:10px'>Rating: {actual_rating:.2f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("")
        if pred == actual:
            st.success("✅ Model prediction matches the actual rating!")
        else:
            st.warning("⚠️ Model prediction did not match the actual rating for this restaurant.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Area Clustering  (fetches from area_df saved from notebook)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Area Clustering":
    st.title("Bangalore Area Market Zones")
    st.markdown("K-Means clustering (K=4) — values fetched from clustering results.")
    st.markdown("")

    zones = ["Growth Opportunity", "Saturated Market", "Underserved Area", "High Competition Low Demand"]
    zone_descs = {
        "Growth Opportunity":          "Low competition, high demand. Best for new entrants.",
        "Saturated Market":            "High density, high demand. Hard to differentiate.",
        "Underserved Area":            "Low density, low demand. Niche first-mover play.",
        "High Competition Low Demand": "High competition, low demand. Risky — avoid.",
    }
    cols = st.columns(4)
    for i, zone in enumerate(zones):
        color, bg = ZONE_STYLE.get(zone, ("#888", "#F8F8F8"))
        count = len(area_df[area_df['market_zone'] == zone])
        with cols[i]:
            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {color}; border-radius:12px; padding:1rem; text-align:center;'>
                <div style='font-size:12px; font-weight:600; color:{color};'>{zone}</div>
                <div style='font-size:28px; font-weight:700; color:#1A1A1A; margin:6px 0'>{count}</div>
                <div style='font-size:11px; color:#777;'>{zone_descs[zone]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#### 🔍 Look up an area")
        area_list = sorted(area_df['area'].tolist())
        selected_area = st.selectbox("Select area", area_list)

        if selected_area:
            row = area_df[area_df['area'] == selected_area].iloc[0]
            zone = row['market_zone']
            color, bg = ZONE_STYLE.get(zone, ("#888", "#F8F8F8"))

            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {color}; border-radius:12px; padding:1.2rem; margin-top:1rem;'>
                <div style='font-size:12px; color:#888; margin-bottom:4px;'>Market Zone</div>
                <div style='font-size:18px; font-weight:600; color:{color};'>{zone}</div>
                <hr style='border-color:#eee; margin:10px 0;'>
                <div style='font-size:13px; color:#555; line-height:2.2;'>
                    🏪 Restaurants in area: <b>{int(row['restaurant_count'])}</b><br>
                    ⭐ Avg Delivery Rating: <b>{row['avg_delivery_rating']:.2f}</b><br>
                    📈 Avg Demand Score: <b>{row['avg_demand_score']:,.0f}</b><br>
                    🎯 Avg Opportunity Score: <b>{row['avg_opportunity']:.1f}</b><br>
                    💰 Avg Cost for 2: <b>₹{row['avg_cost']:,.0f}</b>
                </div>
            </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Cluster scatter — Demand vs Competition")
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors_map = {
            "Growth Opportunity":          "#2ca02c",
            "Saturated Market":            "#d62728",
            "Underserved Area":            "#e6b800",
            "High Competition Low Demand": "#ff7f0e",
        }
        for zone, grp in area_df.groupby('market_zone'):
            ax.scatter(grp['restaurant_count'], grp['avg_demand_score'],
                       c=colors_map.get(zone, '#888'), label=zone,
                       alpha=0.8, s=70, edgecolors='white', linewidths=0.5)
        ax.set_xlabel("Number of Restaurants (Competition)", fontsize=11)
        ax.set_ylabel("Average Demand Score", fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#FAFAF8')
        fig.patch.set_facecolor('#FAFAF8')
        ax.legend(fontsize=8, framealpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### 📋 All areas with zone classification")
    display_df = area_df[['area','restaurant_count','avg_delivery_rating',
                           'avg_demand_score','avg_opportunity','avg_cost','market_zone']].copy()
    display_df.columns = ['Area','Restaurants','Avg Rating','Demand Score','Opportunity Score','Avg Cost','Market Zone']
    display_df = display_df.sort_values('Demand Score', ascending=False).reset_index(drop=True)
    display_df['Avg Rating']        = display_df['Avg Rating'].round(2)
    display_df['Demand Score']      = display_df['Demand Score'].round(0).astype(int)
    display_df['Opportunity Score'] = display_df['Opportunity Score'].round(1)
    display_df['Avg Cost']          = display_df['Avg Cost'].round(0).astype(int)
    st.dataframe(display_df, use_container_width=True, height=400)

    st.markdown("""
    <div style='background:#EDFAF3; border-left:3px solid #2ca02c; padding:0.8rem 1rem; border-radius:6px; font-size:14px; margin-top:1rem;'>
    <b>Business insight:</b> <b>Growth Opportunity</b> areas have low competition but strong demand — best risk-adjusted expansion zones.
    <b>Underserved</b> areas are first-mover opportunities but come with demand risk.
    </div>""", unsafe_allow_html=True)