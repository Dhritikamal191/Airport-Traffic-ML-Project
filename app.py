# ===============================
# IMPORTS
# ===============================
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
st.markdown("""
<style>

/* ===============================
   🌙 MAIN BACKGROUND
=============================== */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
}

/* ===============================
   📌 SIDEBAR
=============================== */
section[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

/* ===============================
   🔤 FORCE INPUT STYLING (KEY FIX)
=============================== */

/* Text + Number Inputs */
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
}

/* Selectbox */
div[data-baseweb="select"] > div {
    background-color: #1f2937 !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

/* Selected value text */
div[data-baseweb="select"] span {
    color: #ffffff !important;
}

/* Dropdown menu */
ul[role="listbox"] {
    background-color: #ffffff !important;
}

/* Dropdown items */
ul[role="listbox"] li {
    color: #000000 !important;
}

/* ===============================
   🏷 LABELS (VERY IMPORTANT)
=============================== */
label, .stSelectbox label, .stNumberInput label {
    color: #e5e7eb !important;
}

/* ===============================
   🎚 SLIDER
=============================== */
.stSlider label {
    color: #e5e7eb !important;
}

/* ===============================
   📊 METRIC TEXT FIX
=============================== */
[data-testid="stMetric"] * {
    color: #f9fafb !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   🔘 BUTTON FIX (VISIBLE + PREMIUM)
=============================== */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.6em 1.2em !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.4);
}

.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   🧾 HEADINGS FIX
=============================== */
h1, h2, h3 {
    color: #f9fafb !important;
    font-weight: 600;
}

/* Subtext */
p, span {
    color: #d1d5db !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   📦 CARD SYSTEM
=============================== */
.card {
    background: #1f2937;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #374151;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   📐 SPACING FIX
=============================== */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* ===============================
   🔝 REMOVE WHITE TOP BAR
=============================== */

/* Header background */
[data-testid="stHeader"] {
    background: #0e1117 !important;
}

/* Toolbar (settings/share icons area) */
[data-testid="stToolbar"] {
    background: #0e1117 !important;
}

/* Entire top block */
[data-testid="stDecoration"] {
    background: #0e1117 !important;
}

/* Optional: hide header completely */
[data-testid="stHeader"] {
    visibility: visible;   /* change to hidden if you want to remove it */
}

/* Remove shadow line */
header {
    box-shadow: none !important;
}

</style>
""", unsafe_allow_html=True)


import streamlit as st

st.markdown(
    """
    <style>
    /* 1. Target the floating menu container */
    div[data-baseweb="popover"] ul {
        background-color: #1E1E1E !important; /* Dark background for the list */
        border: 1px solid #50C878 !important; /* Emerald border */
    }

    /* 2. Style the individual list items (the ICAO codes) */
    div[data-baseweb="popover"] li {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important; /* Force text to white */
    }

    /* 3. Style the item when you hover over it */
    div[data-baseweb="popover"] li:hover {
        background-color: #50C878 !important; /* Emerald highlight */
        color: #000000 !important; /* Black text on hover for contrast */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* 1. REMOVE THE RED BORDER on the selectbox when clicked */
    div[data-baseweb="select"] > div {
        border-color: #50C878 !important; /* Forces your emerald green */
        box-shadow: none !important;
        outline: none !important;
    }

    /* 2. TARGET THE DROPDOWN LIST (The Popover) */
    /* This handles the background of the entire floating menu */
    div[data-baseweb="popover"] > div {
        background-color: #1E1E1E !important;
        border: 1px solid #50C878 !important;
    }

    /* 3. STYLE THE LIST ITEMS (ICAO Codes) */
    div[data-baseweb="popover"] li {
        background-color: #1E1E1E !important;
        color: white !important;
    }

    /* 4. STYLE THE HOVER STATE (When you move your mouse over codes) */
    div[data-baseweb="popover"] li:hover {
        background-color: #50C878 !important;
        color: black !important;
    }

    /* 6. Fix the white background on Expanders (Airport Info, etc.) */
    div[data-testid="stExpander"] details summary {
        background-color: #1E1E1E !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("""
<style>

.header-box {
    background: linear-gradient(
        135deg,
        #0f172a,
        #1e293b,
        #2563eb
    );
    
    padding: 30px;
    border-radius: 18px;

    text-align: center;

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 8px 25px rgba(0,0,0,0.35);

    margin-bottom: 25px;
}

.header-title {
    color: white;
    font-size: 38px;
    font-weight: 800;
    margin-bottom: 10px;
}

.header-subtitle {
    color: #cbd5e1;
    font-size: 16px;
    font-weight: 400;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.kpi-card {
    background: linear-gradient(
        145deg,
        rgba(30,41,59,0.95),
        rgba(15,23,42,0.95)
    );

    padding: 20px;
    border-radius: 16px;

    border: 1px solid rgba(255,255,255,0.08);

    box-shadow:
        0 4px 15px rgba(0,0,0,0.30);

    transition: all 0.3s ease;

    text-align: center;
}

.kpi-card:hover {
    transform: translateY(-4px);

    box-shadow:
        0 8px 20px rgba(37,99,235,0.25);
}

.kpi-title {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
}

.kpi-value {
    color: white;
    font-size: 32px;
    font-weight: 800;
    margin-top: 8px;
}

.kpi-delta-positive {
    color: #22c55e;
    font-size: 14px;
    font-weight: 600;
}

.kpi-delta-negative {
    color: #ef4444;
    font-size: 14px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Airport Traffic Dashboard", layout="wide")

st.markdown("""
<div class="header-box">
    <div class="header-title">
        ✈️ Airport Traffic Forecasting & Analytics
    </div>    
</div>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("xgb_airport_pipeline.pkl", "rb") as f:
        return pickle.load(f)

with open("metrics.json", "r") as f:
    metrics = json.load(f)

mae = metrics["mae"]
rmse = metrics["rmse"]
r2 = metrics["r2"]

model = load_model()

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("airport_traffic_2025.csv")
    df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'])

    # Feature engineering
    df['YEAR'] = df['FLT_DATE'].dt.year
    df['MONTH'] = df['FLT_DATE'].dt.month
    df['DAY'] = df['FLT_DATE'].dt.day
    df['WEEKDAY'] = df['FLT_DATE'].dt.weekday
    df['IS_WEEKEND'] = (df['WEEKDAY'] >= 5).astype(int)

    df['DEP_ARR_RATIO'] = df['FLT_DEP_1'] / (df['FLT_ARR_1'] + 1)
    df['IFR_RATIO'] = df['FLT_TOT_IFR_2'] / (df['FLT_TOT_1'] + 1)

    return df

df = load_data()

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Input Parameters")
# ===============================
# Date Controls
# ===============================
with st.sidebar.expander("Date Controls", expanded=True):
    year = st.selectbox("Year", sorted(df['YEAR'].unique()))
    month = st.selectbox("Month", sorted(df['MONTH'].unique()))
    day = st.slider("Day", 1, 31, 15)

# ===============================
# Airport Selection
# ===============================
with st.sidebar.expander("Airport Info"):
    airport = st.selectbox("Airport (ICAO)", df['APT_ICAO'].unique())
    state = st.selectbox("State", df['STATE_NAME'].unique())

# ===============================
# Feature Controls
# ===============================
with st.sidebar.expander("Model Inputs"):
    weekday = st.selectbox("Weekday (0=Mon)", list(range(7)))
    is_weekend = 1 if weekday >= 5 else 0

    dep_arr_ratio = st.slider("Dep/Arr Ratio", 0.1, 3.0, 0.1)
    ifr_ratio = st.slider("IFR Ratio", 0.0, 1.0, 0.5)

input_df = pd.DataFrame({
    'YEAR':[year],'MONTH':[month],'DAY':[day],'WEEKDAY':[weekday],
    'IS_WEEKEND':[is_weekend],'APT_ICAO':[airport],'STATE_NAME':[state],
    'DEP_ARR_RATIO':[dep_arr_ratio],'IFR_RATIO':[ifr_ratio]
})

# ===============================
# PREDICTION + KPI
# ===============================

filtered_df = df.copy()

# State Filter
if state != "All":
   filtered_df = filtered_df[
        filtered_df["STATE_NAME"] == state
   ]

if day != "All":
   filtered_df = filtered_df [filtered_df["DAY"] == day
   ]

st.subheader("Prediction")

if st.button("Predict Traffic"):
     pred = model.predict(input_df)[0]
     total_flights = filtered_df["FLT_TOT_1"].sum()
     total_ifr = filtered_df["FLT_TOT_IFR_2"].sum()
     active_airports = filtered_df["APT_ICAO"].nunique()

     col1, col2, col3, col4 = st.columns(4)
    
     with col1:
          st.markdown(f"""
          <div class="kpi-card">
           <div class="kpi-title">Total Flights</div>
           <div class="kpi-value">{total_flights:,.0f}</div>
           <div style="color:{color};font-weight:600;">
        {arrow} {abs(delta_pct):.1f}%
           </div>
          </div>
          """, unsafe_allow_html=True)

     with col2:
          st.markdown(f"""
          <div class="kpi-card">
           <div class="kpi-title">Predicted Flights</div>
           <div class="kpi-value">{int(pred)}</div>
           <div style="color:{color};font-weight:600;">
        {arrow} {abs(delta_pct):.1f}%
           </div>
          </div>
          """, unsafe_allow_html=True)

     with col3:
          st.markdown(f"""
          <div class="kpi-card">
           <div class="kpi-title">Active Airports</div>
           <div class="kpi-value">{active_airports}</div>
           <div style="color:{color};font-weight:600;">
        {arrow} {abs(delta_pct):.1f}%
           </div>
          </div>
          """, unsafe_allow_html=True)

     with col4:
          st.markdown(f"""
          <div class="kpi-card">
           <div class="kpi-title">Total IFR Flights</div>
           <div class="kpi-value">{total_ifr}</div>
           <div style="color:{color};font-weight:600;">
        {arrow} {abs(delta_pct):.1f}%
           </div>
          </div>
          """, unsafe_allow_html=True)     
    
# ===============================
# INSIGHTS
# ===============================
tab1, tab2, tab3, tab4, tab5, tab6=st.tabs(["Traffic Insights","Model Explanation","Future Forecast","Traffic Scenarios","Monitoring","Drift"])
with tab1:
     st.subheader("Traffic Insights")
     col1, col2 = st.columns(2)
     with col1:     
          monthly = filtered_df.groupby('MONTH')['FLT_TOT_1'].mean().reset_index()
          fig = px.line(monthly,x='MONTH',y='FLT_TOT_1', markers=True)
          fig.update_layout(xaxis_title="Month",yaxis_title="Monthly Average Flight Traffic",template="plotly_dark",title=dict(text="Monthly Average Traffic",x=0.5, xanchor="center",font=dict(size=17, color="white")),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)
     with col2:     
          top_airports = (filtered_df.groupby('APT_ICAO')['FLT_TOT_1'].sum().nlargest(10).reset_index())
          fig = px.bar(top_airports,x='APT_ICAO',y='FLT_TOT_1',text_auto=True)
          fig.update_layout(xaxis_title="Airport",yaxis_title="Total Flights",title=dict(text="Top 10 Busiest Airports",x=0.5, xanchor="center",font=dict(size=17, color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)

     peak_month = monthly.loc[monthly['FLT_TOT_1'].idxmax(), 'MONTH']
     peak_traffic = monthly['FLT_TOT_1'].max()

     low_month = monthly.loc[monthly['FLT_TOT_1'].idxmin(), 'MONTH']
     low_traffic = monthly['FLT_TOT_1'].min()

     st.info(
      f"""
    📈 Peak traffic occurred in **Month {peak_month}**
    with **{peak_traffic:,.0f} flights**.

    📉 Lowest traffic occurred in **Month {low_month}**
    with **{low_traffic:,.0f} flights**.
    """
     )

     top_airport = top_airports.iloc[0]['APT_ICAO']
     top_volume = top_airports.iloc[0]['FLT_TOT_1']

     st.success(
      f"""
         ✈️ **{top_airport}** handled the highest traffic volume
    with **{top_volume:,.0f} total flights**.
     """
     )
     # ===============================
     # SHAP EXPLANATION
     # ===============================
with tab2:
     st.subheader("Model Explanation")

     if "show_shap" not in st.session_state:
        st.session_state.show_shap = False

     if st.button("Show SHAP Analysis"):
        st.session_state.show_shap = True

     if st.session_state.show_shap:

        try:

            pre = model.named_steps["preprocessor"]
            xgb = model.named_steps["model"]
  
            # ===============================
            # TRANSFORM DATA
            # ===============================
            X_trans = pre.transform(input_df)
            X_sample = pre.transform(df.sample(min(200, len(df)), random_state=42))

            # Convert sparse matrix -> dense
            if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()

            if hasattr(X_sample, "toarray"):
               X_sample = X_sample.toarray()

               X_trans = np.asarray(X_trans, dtype=float)
               X_sample = np.asarray(X_sample, dtype=float)

            feature_names = pre.get_feature_names_out()

            # ===============================
            # SHAP EXPLAINER
            # ===============================
            explainer = shap.TreeExplainer(xgb)

            shap_values = explainer.shap_values(X_trans)
            shap_values_global = explainer.shap_values(X_sample)

            shap_values = np.array(shap_values)
            shap_values_global = np.array(shap_values_global)

            # ===============================
            # GLOBAL FEATURE IMPORTANCE
            # ===============================
            st.subheader("Global SHAP Feature Importance")

            shap_importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(shap_values_global).mean(axis=0)
        }).sort_values("Importance", ascending=False)

            fig_global = px.bar(
            shap_importance.head(15),
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues"
            )
   
            fig_global.update_layout(
            title="Top Features (Global Impact)",
            yaxis={'categoryorder': 'total ascending'},
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_global, use_container_width=True)

            # ===============================
            # LOCAL EXPLANATION
            # ===============================
            st.subheader("Local Explanation")

            local_vals = shap_values[0]

            waterfall_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": local_vals
            })

            waterfall_df = waterfall_df.reindex(
            waterfall_df["SHAP Value"].abs()
            .sort_values(ascending=False).index
        ).head(10)

            fig_waterfall = px.bar(
            waterfall_df,
            x="SHAP Value",
            y="Feature",
            orientation="h",
            color="SHAP Value",
            color_continuous_scale="RdBu"
            )

            fig_waterfall.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_waterfall, use_container_width=True)

            # ===============================
            # INSIGHT
            # ===============================
            top_feature = shap_importance.iloc[0]["Feature"]

            st.info(
            f"🤖 The most influential feature in the current prediction is **{top_feature}**."
        )

            # ===============================
            # DEPENDENCE PLOT
            # ===============================
            st.subheader("Feature Dependence")

            interaction_feature = st.selectbox(
            "Select Feature",
            feature_names
            )
  
            feature_index = list(feature_names).index(
            interaction_feature
            )

            dependence_df = pd.DataFrame({
            "Feature Value": X_sample[:, feature_index],
            "SHAP Value": shap_values_global[:, feature_index]
            })

            fig_dep = px.scatter(
            dependence_df,
            x="Feature Value",
            y="SHAP Value",
            color="SHAP Value",
            color_continuous_scale="Viridis",
            trendline="lowess"
            )
 
            fig_dep.update_layout(
            title=f"Dependence Plot: {interaction_feature}",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
            )
 
            fig_dep.update_xaxes (showgrid=False)
            fig_dep.update_yaxes (showgrid=False)

            st.plotly_chart(fig_dep, use_container_width=True)

        except Exception as e:

                st.error(f"SHAP failed: {str(e)}")

        st.info(
            "Model prediction still works normally. SHAP explanation is temporarily unavailable."
        )
     # ===============================
     # NEXT 6 MONTHS FORECAST
     # ===============================
with tab3:
     import datetime
     st.subheader("Next 6 Months Forecast")
     if st.button("Generate Forecast"):
        # Start from selected year/month
        start_date = datetime.date(int(year), int(month), 1)
        future_data = []
        for i in range(6):
            future_month = (start_date.month + i - 1) % 12 + 1
            future_year = start_date.year + ((start_date.month + i - 1) // 12)
            # Assume mid-month day
            day = 15
            date_obj = datetime.date(future_year, future_month, day)
            weekday = date_obj.weekday()
            is_weekend = 1 if weekday >= 5 else 0
            future_data.append({
            'YEAR': future_year,
            'MONTH': future_month,
            'DAY': day,
            'WEEKDAY': weekday,
            'IS_WEEKEND': is_weekend,
            'APT_ICAO': airport,
            'STATE_NAME': state,
            'DEP_ARR_RATIO': dep_arr_ratio,
            'IFR_RATIO': ifr_ratio
            })
            future_df = pd.DataFrame(future_data)
            # Predict
            predictions = model.predict(future_df)
            future_df['Predicted Flights'] = predictions
        # ===============================
        # PLOT
        # ===============================
        fig = px.line(future_df,x='MONTH',y='Predicted Flights',markers=True,title="Next 6 Months Flight Forecast")
        fig.update_layout(title=dict(text="Next 6 Months Flight Forecast",x=0.5, xanchor="center",font=dict(size=17, color="white")),xaxis_title="Month",yaxis_title= "Predicted Flights",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        # ===============================
        # TABLE
        # ===============================
        st.dataframe(future_df[['YEAR','MONTH','Predicted Flights']])
        st.subheader("IFR vs Total Flights Comparison")
        ifr_df = df.groupby(['APT_ICAO','MONTH'])[['FLT_TOT_1','FLT_TOT_IFR_2']].mean().reset_index()
        fig = px.scatter(ifr_df,x='MONTH',y=['FLT_TOT_1','FLT_TOT_IFR_2'],color_continuous_scale="Plasma")
        fig.update_traces(marker=dict(size=15, line=dict(width=0,color="rgba(255,255,255,0.4)"))) 
        fig.update_layout(title=dict(text="IFR vs Total Flights (Monthly)",x=0.5, xanchor="center",font=dict(size=17, color="white")),legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

        growth = (
    (future_df['Predicted Flights'].iloc[-1]
     - future_df['Predicted Flights'].iloc[0])
    /
    future_df['Predicted Flights'].iloc[0]
) * 100

        if growth > 0:
           st.success(
           f"📈 Forecast suggests traffic may increase by {growth:.1f}% over the next 6 months.")
        else:
             st.warning(
             f"📉 Forecast suggests traffic may decrease by {abs(growth):.1f}% over the next 6 months.")

with tab4:
     col1,col2=st.columns(2)
     with col1:
          state_df = df.groupby('STATE_NAME')['FLT_TOT_1'].sum().nlargest(10).reset_index()
          fig = px.bar(state_df,x='STATE_NAME',y='FLT_TOT_1',text_auto=True)
          fig.update_layout(title=dict(text="Top States by Traffic",x=0.5, xanchor="center",font=dict(size=17, color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
          fig.update_xaxes(showgrid=False)
          fig.update_yaxes(showgrid=False)
          st.plotly_chart(fig, use_container_width=True)
     with col2:
          week_df = filtered_df.groupby('IS_WEEKEND')['FLT_TOT_1'].mean().reset_index()
          week_df['Type'] = week_df['IS_WEEKEND'].map({0: 'Weekday', 1: 'Weekend'})
          fig = px.pie(week_df,names='Type',values='FLT_TOT_1',hole=0.5)
          fig.update_traces(textinfo='percent+label',hovertemplate="<b>%{label}</b><br>Flights: %{value:.0f}<br>Share: %{percent}")
          fig.update_layout(title=dict(text="Weekend vs Weekday Traffic",x=0.5,xanchor="center",font=dict(size=17, color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",legend=dict(font=dict(color="white"),orientation="h",y=-0.1))
          st.plotly_chart(fig, use_container_width=True)
     
     heat_df = filtered_df.pivot_table(values='FLT_TOT_1',index='MONTH', columns='DAY',aggfunc='mean')
     fig = px.imshow(heat_df,aspect="auto",title="Traffic Heatmap (Month vs Day)")
     fig.update_layout(title=dict(text="Traffic Heat Map (Year vs Month)",x=0.5, xanchor="center",font=dict(size=17, color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig, use_container_width=True)
     
     X = df[['YEAR', 'MONTH', 'DAY', 'WEEKDAY', 'IS_WEEKEND','APT_ICAO', 'STATE_NAME','DEP_ARR_RATIO', 'IFR_RATIO']]
     y = df['FLT_TOT_1']    
     df['Predicted'] = model.predict(X)
     actual_vs_pred_df = df[['FLT_DATE', 'FLT_TOT_1', 'Predicted']].copy()
     actual_vs_pred_df.rename(columns={'FLT_TOT_1': 'Actual'}, inplace=True)
     fig = px.scatter(actual_vs_pred_df,x='Actual',y='Predicted',trendline="ols",opacity=0.6)
     min_val = min(actual_vs_pred_df['Actual'].min(), actual_vs_pred_df['Predicted'].min())
     max_val = max(actual_vs_pred_df['Actual'].max(), actual_vs_pred_df['Predicted'].max())
     fig.add_shape(type="line",x0=min_val, y0=min_val,x1=max_val, y1=max_val,line=dict(dash="dash"))
     fig.update_layout(title=dict(text="Actual vs Predicted (Model Performance)",x=0.5, xanchor="center",font=dict(size=17, color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Actual Flights",yaxis_title="Predicted Flights",title_x=0.3)
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     fig.update_traces(marker=dict(size=6),hovertemplate="<b>Actual:</b> %{x}<br>" +"<b>Predicted:</b> %{y}<extra></extra>")
     st.plotly_chart(fig, use_container_width=True)

with tab5:

     reference = filtered_df.iloc[:len(filtered_df)//2]
     current = filtered_df.iloc[len(filtered_df)//2:]

     ref_total = reference["FLT_TOT_1"].sum()
     curr_total = current["FLT_TOT_1"].sum()

     if ref_total != 0:
        delta_pct = ((curr_total - ref_total) / ref_total) * 100
     else:
          delta_pct = 0
 
     if delta_pct >= 0:
        arrow = "▲"
        color = "#22c55e"
     else:
          arrow = "▼"
          color = "#ef4444"

     col1, col2, col3 = st.columns(3)

     with col1:
          st.markdown(f"""
              <div class="kpi-card">
               <div class="kpi-title">MAE</div>
               <div class= "kpi-value"> {mae:,.2f}
              </div>
              """, unsafe_allow_html=True)
     
          st.info("Average flights the prediction differs from actual traffic.")

     with col2:
          st.markdown(f"""
              <div class="kpi-card">
               <div class="kpi-title">RMSE</div>
               <div class= "kpi-value"> {rmse:,.2f}
              </div>
              """, unsafe_allow_html=True)    
          st.info("Measures prediction error with higher penalty for large mistakes.")

     with col3:
          st.markdown(f"""
              <div class="kpi-card">
               <div class="kpi-title">R² Score</div>
               <div class= "kpi-value"> {r2:,.3f}
              </div>
              """, unsafe_allow_html=True)
   
          st.info("Percentage of traffic variability explained by the model.")

     fig = px.line(actual_vs_pred_df,x='FLT_DATE',y=['Actual', 'Predicted'])
     fig.update_layout(title=dict(text="Prediction Monitoring Over Time",x=0.5, xanchor="center",font=dict(size=17, color="white")),legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

     actual_vs_pred_df['Error'] = abs(actual_vs_pred_df['Actual'] - actual_vs_pred_df['Predicted'])
     error_trend = actual_vs_pred_df.groupby(actual_vs_pred_df['FLT_DATE'].dt.month)['Error'].mean().reset_index()
     actual_vs_pred_df['Rolling_Error'] = actual_vs_pred_df['Error'].rolling(5).mean()
     fig = px.line(actual_vs_pred_df,x='FLT_DATE',y=['Error', 'Rolling_Error'],title="Error Trend Monitoring")
     threshold = actual_vs_pred_df['Error'].mean() * 1.5
     fig.add_hline(y=threshold,line_dash="dash",annotation_text="Alert Threshold")
     fig.update_layout(title=dict(text="Error Trend Monitoring",x=0.5,xanchor="center",font=dict(size=17, color="white")),legend=dict(font=dict(color="white")),template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Date",yaxis_title="Error")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

with tab6:
     filtered_df = filtered_df.sort_values("FLT_DATE")
     split_idx = len(filtered_df) // 2

     reference = filtered_df.iloc[:split_idx]
     current = filtered_df.iloc[split_idx:]

     ref_avg = reference['FLT_TOT_1'].mean()
     curr_avg = current['FLT_TOT_1'].mean()

     if (
        pd.isna(ref_avg)
        or pd.isna(curr_avg)
        or ref_avg == 0
        ):
          drift_pct = 0
     else:
          drift_pct = ((curr_avg - ref_avg) / ref_avg) * 100

     st.metric(
     "Traffic Drift %",
     f"{drift_pct:.2f}%",
     delta=f"{drift_pct:.2f}%"
     )
     airport_dist = (filtered_df.groupby(['MONTH','APT_NAME'])['FLT_TOT_1'].sum().reset_index())
     fig = px.scatter(airport_dist,x='MONTH',y='FLT_TOT_1',color='APT_NAME',title="Airport Traffic Distribution Drift", color_continuous_scale="Turbo")
     fig.update_traces(marker=dict(size=15, line=dict(width=0,color="rgba(255,255,255,0.4)")))
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)") 
     st.plotly_chart(fig, use_container_width=True)
     state_drift = (df.groupby('STATE_NAME')['FLT_TOT_1'].mean().sort_values(ascending=False).head(15))
     fig = px.line(state_drift,title="State Traffic Drift")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     st.plotly_chart(fig, use_container_width=True)
     filtered_df['IFR_RATIO'] = (filtered_df['FLT_TOT_IFR_2'] / filtered_df['FLT_TOT_1']+1)  
     ref_ifr = reference['IFR_RATIO'].mean()
     curr_ifr = current['IFR_RATIO'].mean()

     if pd.isna(ref_ifr) or pd.isna(curr_ifr):
        ifr_drift = 0
        curr_ifr = 0
     else:
          ifr_drift = curr_ifr - ref_ifr

     st.metric(
     "IFR Ratio Drift",
     f"{curr_ifr:.2%}",
     delta=f"{ifr_drift:.2%}"
     )
     ifr_trend= (filtered_df.groupby("MONTH")["IFR_RATIO"].mean().reset_index())
     fig = px.scatter(ifr_trend,x='MONTH',y='IFR_RATIO',color='MONTH',title="Growth by Month", color_continuous_scale="Plasma")
     fig.update_traces(marker=dict(size=40, line=dict(width=0,color="rgba(255,255,255,0.4)"))) 
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")     
     st.plotly_chart(fig, use_container_width=True)       
     if abs(drift_pct) < 5:
        risk = "🟢 Stable"
     elif abs(drift_pct) < 15:
          risk = "🟡 Moderate Drift"
     else:
          risk = "🔴 Significant Drift"
     
     st.subheader("Quantitative Analysis")  
     monthly = (
     df.groupby('MONTH')['FLT_TOT_1']
      .sum()
      .reset_index()
     )

     monthly['Growth_%'] = monthly['FLT_TOT_1'].pct_change() * 100

     airport_share = (
     filtered_df.groupby('APT_ICAO')['FLT_TOT_1']
      .sum()
      .reset_index()
     )

     airport_share['Market Share %'] = (
     airport_share['FLT_TOT_1']
     / airport_share['FLT_TOT_1'].sum()
     ) * 100

     filtered_df['IFR_RATIO'] = (
     filtered_df['FLT_TOT_IFR_2']
     / (filtered_df['FLT_TOT_1'] + 1)
     )

     filtered_df['EFFICIENCY_SCORE'] = (
     filtered_df['FLT_ARR_1']
     / (filtered_df['FLT_DEP_1'] + 1)
     )

     corr_cols = [
     'FLT_DEP_1',
     'FLT_ARR_1',
     'FLT_DEP_IFR_2',
     'FLT_ARR_IFR_2',
     'FLT_TOT_1'
     ]

     corr = filtered_df[corr_cols].corr()     

     corr_cols = [
     'FLT_DEP_1',
     'FLT_ARR_1',
     'FLT_TOT_1',
     'FLT_DEP_IFR_2',
     'FLT_ARR_IFR_2',
     'FLT_TOT_IFR_2',
     'DEP_ARR_RATIO',
     'IFR_RATIO'
     ]

     corr_matrix = filtered_df[corr_cols].corr().round(2)

     fig = px.imshow(
     corr_matrix,
     text_auto=True,
     color_continuous_scale="Turbo",
     zmin=-1,
     zmax=1,
     aspect="auto"
     )

     fig.update_layout(
     title={
        "text":"📊 Flight Operations Correlation Matrix",
        "x":0.5,
        "font":{"size":22}
     },
     template="plotly_dark",
     height=750,
     coloraxis_colorbar=dict(
        title="Correlation"
     ),
     paper_bgcolor="rgba(0,0,0,0)",
     plot_bgcolor="rgba(0,0,0,0)"
     )
     
     st.plotly_chart(fig,use_container_width=True)

     corr_pairs = (
     corr_matrix.abs()
     .unstack()
     .sort_values(ascending=False)
     )
     corr_pairs = corr_pairs[corr_pairs < 1]
     top_corr = corr_pairs.head(1)

     st.info(
     f"""
     🔍 Strongest relationship detected:
     **{top_corr.index[0][0]}** and
     **{top_corr.index[0][1]}**
     with correlation **{top_corr.iloc[0]:.2f}**
     """
     )