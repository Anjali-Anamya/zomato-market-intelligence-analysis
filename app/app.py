import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Zomato Market Intelligence",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #FAFAF8; }

.metric-card {
    background: white;
    border: 1px solid #EBEBEB;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}
.metric-card .label { font-size: 12px; color: #888; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.metric-card .value { font-size: 28px; font-weight: 600; color: #1A1A1A; }
.metric-card .sub { font-size: 12px; color: #aaa; margin-top: 2px; }

.predict-box {
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1rem;
}
.success-box { background: #EDFAF3; border: 1.5px solid #34C77B; }
.fail-box    { background: #FEF2F2; border: 1.5px solid #F87171; }

.zone-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 500;
}
.sidebar-section {
    background: #F5F5F2;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf      = joblib.load("models/rf_model.pkl")
    km      = joblib.load("models/kmeans_model.pkl")
    scaler  = joblib.load("models/scaler.pkl")
    area_df = joblib.load("models/area_df.pkl")
    return rf, km, scaler, area_df

rf, km, scaler, area_df = load_models()

FEATURES = ['dinner_ratings','dinner_reviews','delivery_reviews','averagecost',
            'ishomedelivery','istakeaway','isindoorseating','isvegonly',
            'cuisine_count','restaurant_density','cost_index',
            'demand_score','opportunity_score','area_avg_rating']

CLUSTER_LABELS = {
    0: ("Growth Opportunity",  "#22C55E", "#EDFAF3"),
    1: ("Saturated",           "#F59E0B", "#FEF9EC"),
    2: ("Underserved",         "#6366F1", "#F0F0FF"),
    3: ("High Competition\nLow Demand", "#EF4444", "#FEF2F2"),
}

# ── Sidebar navigation ───────────────────────────────────────────────────────
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
            <div class='value'>88.4%</div>
            <div class='sub'>Random Forest</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class='metric-card'>
            <div class='label'>Areas Segmented</div>
            <div class='value'>147</div>
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
        Input restaurant attributes and get an instant prediction on whether a restaurant
        will be **high-rated (≥4.0)** or **low-rated**.

        - **Model**: Random Forest Classifier
        - **Features**: 14 engineered inputs
        - **Metric**: Accuracy 88.4%, F1 0.775
        """)

    with col2:
        st.markdown("#### 🗺️ Area Market Zone Classifier")
        st.markdown("""
        Select a Bangalore area to see which **market zone** it falls into and
        what that means for business expansion.

        - **Model**: K-Means Clustering (K=4)
        - **Zones**: Growth Opportunity, Saturated, Underserved, High Competition Low Demand
        - **Validated**: Silhouette score
        """)

    st.markdown("---")
    st.markdown("#### 📊 Feature Importance — What drives restaurant success?")

    importances = rf.feature_importances_
    feat_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#6366F1' if v > 0.08 else '#CBD5E1' for v in feat_imp['Importance']]
    bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors, height=0.6)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(left=False)
    ax.set_facecolor('#FAFAF8')
    fig.patch.set_facecolor('#FAFAF8')
    for bar, val in zip(bars, feat_imp['Importance']):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#555')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div style='background:#F0F0FF; border-left: 3px solid #6366F1; padding: 0.8rem 1rem; border-radius: 6px; font-size:14px;'>
    <b>Key insight:</b> <code>dinner_ratings</code>, <code>area_avg_rating</code>, and <code>opportunity_score</code>
    are the top predictors of restaurant success — confirming that location quality and peer ratings
    matter more than price or cuisine variety.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Predict Success
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Predict Success":
    st.title("Restaurant Success Predictor")
    st.markdown("Fill in the restaurant details below to predict if it will be **high-rated (≥ 4.0)**.")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Ratings & Reviews**")
        dinner_ratings   = st.slider("Dinner Rating", 1.0, 5.0, 3.8, 0.1)
        dinner_reviews   = st.number_input("Dinner Reviews", 0, 100000, 500, 50)
        delivery_reviews = st.number_input("Delivery Reviews", 0, 100000, 1000, 100)

    with col2:
        st.markdown("**Restaurant Details**")
        averagecost      = st.number_input("Avg Cost for 2 (₹)", 100, 5000, 500, 50)
        cuisine_count    = st.slider("Number of Cuisines", 1, 15, 3)
        ishomedelivery   = st.selectbox("Home Delivery?",  [1, 0], format_func=lambda x: "Yes" if x else "No")
        istakeaway       = st.selectbox("Takeaway?",       [1, 0], format_func=lambda x: "Yes" if x else "No")
        isindoorseating  = st.selectbox("Indoor Seating?", [1, 0], format_func=lambda x: "Yes" if x else "No")
        isvegonly        = st.selectbox("Veg Only?",       [0, 1], format_func=lambda x: "Yes" if x else "No")

    with col3:
        st.markdown("**Area / Engineered Features**")
        area_avg_rating    = st.slider("Area Avg Rating",    1.0, 5.0, 3.9, 0.05)
        restaurant_density = st.number_input("Restaurant Density (area)", 1, 500, 50)
        demand_score       = st.number_input("Demand Score", 0.0, 100000.0, 5000.0, 100.0)
        opportunity_score  = st.number_input("Opportunity Score", 0.0, 500.0, 80.0, 5.0)
        cost_index         = st.number_input("Cost Index", 0.0, 5.0, 1.0, 0.1)

    st.markdown("")
    predict_btn = st.button("🔍 Predict", use_container_width=True, type="primary")

    if predict_btn:
        input_data = pd.DataFrame([[
            dinner_ratings, dinner_reviews, delivery_reviews, averagecost,
            ishomedelivery, istakeaway, isindoorseating, isvegonly,
            cuisine_count, restaurant_density, cost_index,
            demand_score, opportunity_score, area_avg_rating
        ]], columns=FEATURES)

        pred       = rf.predict(input_data)[0]
        prob       = rf.predict_proba(input_data)[0]
        confidence = prob[pred] * 100

        st.markdown("---")
        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            if pred == 1:
                st.markdown(f"""
                <div class='predict-box success-box'>
                    <div style='font-size:48px'>✅</div>
                    <div style='font-size:22px; font-weight:600; color:#16A34A; margin:8px 0'>High-Rated Restaurant</div>
                    <div style='font-size:14px; color:#555'>Predicted rating ≥ 4.0 — likely to succeed</div>
                    <div style='font-size:28px; font-weight:600; color:#16A34A; margin-top:12px'>{confidence:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='predict-box fail-box'>
                    <div style='font-size:48px'>⚠️</div>
                    <div style='font-size:22px; font-weight:600; color:#DC2626; margin:8px 0'>Low-Rated Restaurant</div>
                    <div style='font-size:14px; color:#555'>Predicted rating &lt; 4.0 — improvement needed</div>
                    <div style='font-size:28px; font-weight:600; color:#DC2626; margin-top:12px'>{confidence:.1f}% confident</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")
        st.markdown("**Probability breakdown**")
        p_col1, p_col2 = st.columns(2)
        p_col1.metric("Low-Rated probability",  f"{prob[0]*100:.1f}%")
        p_col2.metric("High-Rated probability", f"{prob[1]*100:.1f}%")

        st.markdown("---")
        st.markdown("#### 💡 Insight")
        if pred == 1:
            st.success(f"This restaurant profile is predicted to succeed with **{confidence:.1f}% confidence**. "
                       f"Strong dinner ratings and a high area average rating are key contributors.")
        else:
            st.warning(f"This restaurant profile is at risk. Consider improving dinner ratings and "
                       f"targeting a higher-demand area to boost success probability.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Area Clustering
# ════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Area Clustering":
    st.title("Bangalore Area Market Zones")
    st.markdown("K-Means clustering (K=4) segments 147 Bangalore areas into strategic market zones.")
    st.markdown("")

    # Zone summary cards
    zone_data = {
        "Growth Opportunity":              {"color": "#22C55E", "bg": "#EDFAF3", "desc": "High opportunity, moderate density. Best for new entrants.", "areas": len(area_df[area_df['cluster']==0])},
        "Saturated":                       {"color": "#F59E0B", "bg": "#FEF9EC", "desc": "High density, competitive. Hard to differentiate.", "areas": len(area_df[area_df['cluster']==1])},
        "Underserved":                     {"color": "#6366F1", "bg": "#F0F0FF", "desc": "Very low density, high opportunity. Untapped market.", "areas": len(area_df[area_df['cluster']==2])},
        "High Competition Low Demand":     {"color": "#EF4444", "bg": "#FEF2F2", "desc": "Low demand, high competition. Avoid unless differentiated.", "areas": len(area_df[area_df['cluster']==3])},
    }

    cols = st.columns(4)
    for i, (zone, info) in enumerate(zone_data.items()):
        with cols[i]:
            st.markdown(f"""
            <div style='background:{info["bg"]}; border:1.5px solid {info["color"]}; border-radius:12px; padding:1rem; text-align:center;'>
                <div style='font-size:13px; font-weight:600; color:{info["color"]};'>{zone}</div>
                <div style='font-size:28px; font-weight:700; color:#1A1A1A; margin:6px 0'>{info["areas"]}</div>
                <div style='font-size:11px; color:#777;'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("---")

    # Area lookup
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("#### 🔍 Look up an area")
        area_list = sorted(area_df['area'].tolist())
        selected_area = st.selectbox("Select area", area_list)

        if selected_area:
            row = area_df[area_df['area'] == selected_area].iloc[0]
            cluster_id = int(row['cluster'])
            label, color, bg = CLUSTER_LABELS[cluster_id]

            st.markdown(f"""
            <div style='background:{bg}; border:1.5px solid {color}; border-radius:12px; padding:1.2rem; margin-top:1rem;'>
                <div style='font-size:12px; color:#888; margin-bottom:4px;'>Market Zone</div>
                <div style='font-size:18px; font-weight:600; color:{color};'>{label}</div>
                <hr style='border-color:#eee; margin:10px 0;'>
                <div style='font-size:13px; color:#555; line-height:2;'>
                    📍 Area Avg Rating: <b>{row['area_avg_rating']:.2f}</b><br>
                    🏪 Restaurant Density: <b>{int(row['restaurant_density'])}</b><br>
                    📈 Demand Score: <b>{row['demand_score']:,.0f}</b><br>
                    🎯 Opportunity Score: <b>{row['opportunity_score']:.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Cluster scatter — Opportunity vs Density")
        cluster_colors = {0: "#22C55E", 1: "#F59E0B", 2: "#6366F1", 3: "#EF4444"}
        cluster_names  = {0: "Growth Opportunity", 1: "Saturated", 2: "Underserved", 3: "High Competition Low Demand"}

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for c in range(4):
            sub = area_df[area_df['cluster'] == c]
            ax.scatter(sub['restaurant_density'], sub['opportunity_score'],
                       c=cluster_colors[c], label=cluster_names[c],
                       alpha=0.8, s=60, edgecolors='white', linewidths=0.5)
        ax.set_xlabel("Restaurant Density", fontsize=11)
        ax.set_ylabel("Opportunity Score", fontsize=11)
        ax.spines[['top','right']].set_visible(False)
        ax.set_facecolor('#FAFAF8')
        fig.patch.set_facecolor('#FAFAF8')
        ax.legend(fontsize=9, framealpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown("---")
    st.markdown("#### 📋 All areas with zone classification")
    display_df = area_df[['area','area_avg_rating','restaurant_density','demand_score','opportunity_score','cluster']].copy()
    display_df['zone'] = display_df['cluster'].map({k: v[0] for k, v in CLUSTER_LABELS.items()})
    display_df = display_df.drop('cluster', axis=1)
    display_df.columns = ['Area','Avg Rating','Density','Demand Score','Opportunity Score','Zone']
    display_df = display_df.sort_values('Opportunity Score', ascending=False).reset_index(drop=True)
    display_df['Avg Rating'] = display_df['Avg Rating'].round(2)
    display_df['Demand Score'] = display_df['Demand Score'].round(0).astype(int)
    display_df['Opportunity Score'] = display_df['Opportunity Score'].round(1)
    st.dataframe(display_df, use_container_width=True, height=400)

    st.markdown("""
    <div style='background:#F0F0FF; border-left:3px solid #6366F1; padding:0.8rem 1rem; border-radius:6px; font-size:14px; margin-top:1rem;'>
    <b>Business insight:</b> Areas in the <b>Growth Opportunity</b> zone offer the best risk-adjusted expansion potential —
    moderate competition with strong demand signals. <b>Underserved</b> areas are high-risk, high-reward plays for first movers.
    </div>
    """, unsafe_allow_html=True)
